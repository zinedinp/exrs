//! How to read arbitrary but specific selection of arbitrary channels.
//! This is not a zero-cost abstraction.

use std::any::Any;
use std::marker::PhantomData;

use crate::{
    block::{chunk::TileCoordinates, samples::*, UncompressedBlock},
    error::*,
    image::{
        read::layers::{ChannelsReader, ReadChannels},
        recursive::*,
        *,
    },
    io::Read,
    math::*,
    meta::header::*,
};

/// Can be attached one more channel reader.
///
/// Call `required` or `optional` on this object to declare another channel to
/// be read from the file. Call `collect_pixels` at last to define how the
/// previously declared pixels should be stored.
pub trait ReadSpecificChannel: Sized + CheckDuplicates {
    /// A separate internal reader for the pixels. Will be of type `Recursive<_,
    /// SampleReader<_>>`, depending on the pixels of the specific channel
    /// combination.
    type RecursivePixelReader: RecursivePixelReader;

    /// Create a separate internal reader for the pixels of the specific channel
    /// combination.
    fn create_recursive_reader(&self, channels: &ChannelList)
        -> Result<Self::RecursivePixelReader>;

    /// Plan to read an additional channel from the image, with the specified
    /// name. If the channel cannot be found in the image when the image is
    /// read, the image will not be loaded. The generic parameter can
    /// usually be inferred from the closure in `collect_pixels`.
    fn required<Sample>(self, channel_name: impl Into<Text>) -> ReadRequiredChannel<Self, Sample> {
        let channel_name = channel_name.into();
        assert!(
            self.already_contains(&channel_name).not(),
            "a channel with the name `{}` is already defined",
            channel_name
        );
        ReadRequiredChannel {
            channel_name,
            previous_channels: self,
            px: Default::default(),
        }
    }

    /// Plan to read an additional channel from the image, with the specified
    /// name. If the file does not contain this channel, the specified
    /// default sample will be returned instead. You can check whether the
    /// channel has been loaded by checking the presence of the optional
    /// channel description before instantiating your own image. The generic
    /// parameter can usually be inferred from the closure in `collect_pixels`.
    fn optional<Sample>(
        self,
        channel_name: impl Into<Text>,
        default_sample: Sample,
    ) -> ReadOptionalChannel<Self, Sample> {
        let channel_name = channel_name.into();
        assert!(
            self.already_contains(&channel_name).not(),
            "a channel with the name `{}` is already defined",
            channel_name
        );
        ReadOptionalChannel {
            channel_name,
            previous_channels: self,
            default_sample,
        }
    }

    /// Using two closures, define how to store the pixels.
    /// The first closure creates an image, and the second closure inserts a
    /// single pixel. The type of the pixel can be defined by the second
    /// closure; it must be a tuple containing `f16`, `f32`, `u32` or
    /// `Sample` values. See the examples for more information.
    ///
    /// When `PixelStorage` implements [`RowMajorPixelStorage`] with
    /// `Element = Pixel` (for example [`pixel_vec::PixelVec`] or
    /// [`FlatRowMajorPixelStorage`]), the serial reader writes decoded pixels
    /// directly into hoisted row slices and **may not call** `set_pixel`.
    /// Use a non-identity `set_pixel` only with non-row-major storage, or
    /// apply transforms after the read. For element types that differ from
    /// `Pixel`
    fn collect_pixels<Pixel, PixelStorage, CreatePixels, SetPixel>(
        self, create_pixels: CreatePixels, set_pixel: SetPixel
    ) -> CollectPixels<Self, Pixel, PixelStorage, CreatePixels, SetPixel>
        where
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
            CreatePixels: Fn(
                Vec2<usize>,
                &<<Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive
            ) -> PixelStorage,
            SetPixel: Fn(&mut PixelStorage, Vec2<usize>, Pixel),
{
        CollectPixels {
            read_channels: self,
            set_pixel,
            create_pixels,
            px: Default::default(),
        }
    }

    /// Like `collect_pixels`, but writes pixels directly from decompression
    /// worker threads when reading with multiple threads (the default,
    /// unless `.non_parallel()` is used), instead of `collect_pixels`'
    /// single-threaded pixel-storage conversion. Can give a substantial
    /// speedup on multi-core machines, at the cost of a different
    /// `set_row_pixel` closure signature: it receives one row of the pixel
    /// storage as a slice plus an `x` index, rather than the whole storage
    /// plus a `Vec2` position. Requires `PixelStorage` to implement
    /// [`RowMajorPixelStorage`] (implemented by [`FlatRowMajorPixelStorage`]).
    /// A single contiguous buffer is required (rather than one `Vec` per row)
    /// so that scanline workers can split disjoint full-width row bands out
    /// of it. Tiled files also decompress in parallel; their pixel writes
    /// share the flat buffer under a brief lock (tile rectangles may share
    /// rows), still with correct per-tile `x` and width.
    #[cfg(feature = "rayon")]
    fn collect_pixels_in_parallel<Pixel, PixelStorage, CreatePixels, SetRowPixel>(
        self, create_pixels: CreatePixels, set_row_pixel: SetRowPixel
    ) -> CollectPixelsInParallel<Self, Pixel, PixelStorage, CreatePixels, SetRowPixel>
        where
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
            CreatePixels: Fn(
                Vec2<usize>,
                &<<Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive
            ) -> PixelStorage,
            PixelStorage: RowMajorPixelStorage,
            SetRowPixel: Fn(&mut [PixelStorage::Element], usize, Pixel),
    {
        CollectPixelsInParallel {
            read_channels: self,
            set_row_pixel,
            create_pixels,
            px: Default::default(),
        }
    }
}

/// Pixel storage backed by one contiguous, row-major buffer.
///
/// Used by:
/// - the **serial** [`collect_pixels`](ReadSpecificChannel::collect_pixels)
///   path, which hoists one row slice per scanline when
///   `Element = Pixel` (see that method's docs), and
/// - [`collect_pixels_in_parallel`](ReadSpecificChannel::collect_pixels_in_parallel)
///   (requires the `rayon` feature), which splits disjoint row ranges out to
///   decompression worker threads.
///
/// Contiguity matters for the parallel path: scanline workers obtain a row
/// range by splitting one buffer, and tiled workers write through a shared
/// flat view, rather than indexing into separately heap-allocated rows.
pub trait RowMajorPixelStorage {
    /// The element type stored per pixel slot, as passed to `set_row_pixel`.
    /// Not necessarily the same type `set_row_pixel` receives as its `Pixel`
    /// argument -- like `SetPixel` in `collect_pixels`, `set_row_pixel` may
    /// convert its `Pixel` argument to any representation this element type
    /// needs.
    ///
    /// When `Element` is the same type as the decoded `Pixel` passed to
    /// [`collect_pixels`](ReadSpecificChannel::collect_pixels), the serial
    /// reader may write pixels directly into row slices without calling
    /// `set_pixel`.
    type Element;

    /// Width of one row, in elements. Must match the image's pixel width.
    fn width(&self) -> usize;

    /// Mutable access to every pixel as one flat, contiguous, row-major
    /// buffer (row 0 first, then row 1, and so on), `width() * height`
    /// elements long.
    fn pixels_mut(&mut self) -> &mut [Self::Element];
}

/// The reference [`RowMajorPixelStorage`] implementation: a single flat
/// `Vec`, addressed row-major with the given `width`.
#[derive(Clone, Debug)]
pub struct FlatRowMajorPixelStorage<Element> {
    /// Width of one row, in elements. Must match the image's pixel width.
    pub width: usize,

    /// All pixels, row-major, `width * height` elements long.
    pub pixels: Vec<Element>,
}

impl<Element> RowMajorPixelStorage for FlatRowMajorPixelStorage<Element> {
    type Element = Element;

    fn width(&self) -> usize { self.width }

    fn pixels_mut(&mut self) -> &mut [Element] {
        &mut self.pixels
    }
}

impl<T> RowMajorPixelStorage for crate::image::pixel_vec::PixelVec<T> {
    type Element = T;

    #[inline]
    fn width(&self) -> usize {
        self.resolution.width()
    }

    #[inline]
    fn pixels_mut(&mut self) -> &mut [T] {
        &mut self.pixels
    }
}

/// How many pixels of one scanline are decoded before being written out, so
/// that the scratch line, the source bytes and the destination run all stay
/// resident while `read_pixels` makes its one pass per channel over them.
/// Lines narrower than this are still handled in a single run.
const LINE_RUN_PIXELS: usize = 1024;

fn write_decoded_line<PixelStorage: 'static, SetPixel, Pixel: 'static>(
    storage: &mut PixelStorage,
    set_pixel: &SetPixel,
    y: usize,
    x0: usize,
    pixels: impl ExactSizeIterator<Item = Pixel>,
) where
    SetPixel: Fn(&mut PixelStorage, Vec2<usize>, Pixel),
{
    fn write_row<Pixel>(
        flat: &mut [Pixel],
        width: usize,
        y: usize,
        x0: usize,
        pixels: impl ExactSizeIterator<Item = Pixel>,
    ) {
        // Slice down to exactly the pixels this line writes, so the loop below
        // is a straight `zip` over two equal-length sequences: one bounds check
        // per line instead of one per pixel, and no `x0 + i` address arithmetic
        // (the destination is just a walking pointer).
        let start = y * width + x0;
        let row = &mut flat[start..start + pixels.len()];
        for (destination, pixel) in row.iter_mut().zip(pixels) {
            *destination = pixel;
        }
    }

    // Each `downcast_mut` reborrows `storage` only for the duration of the
    // `if let`; when it misses, the original `&mut PixelStorage` binding is
    // free again for the next attempt, or for the closure fallback
    if let Some(flat) =
        (&mut *storage as &mut dyn Any).downcast_mut::<FlatRowMajorPixelStorage<Pixel>>()
    {
        return write_row(&mut flat.pixels, flat.width, y, x0, pixels);
    }
    if let Some(vec) =
        (&mut *storage as &mut dyn Any).downcast_mut::<crate::image::pixel_vec::PixelVec<Pixel>>()
    {
        let width = vec.resolution.width();
        return write_row(&mut vec.pixels, width, y, x0, pixels);
    }

    for (i, pixel) in pixels.enumerate() {
        set_pixel(storage, Vec2(x0 + i, y), pixel);
    }
}

/// A reader containing sub-readers for reading the pixel content of an image.
pub trait RecursivePixelReader {
    /// The channel descriptions from the image.
    /// Will be converted to a tuple before being stored in `SpecificChannels<_,
    /// ChannelDescriptions>`.
    type RecursiveChannelDescriptions;

    /// Returns the channel descriptions based on the channels in the file.
    fn get_descriptions(&self) -> Self::RecursiveChannelDescriptions;

    /// The pixel type. Will be converted to a tuple at the end of the process.
    type RecursivePixel: Copy + Default + 'static;

    /// Read a horizontal run of pixels out of one line of a block.
    ///
    /// `bytes` is the whole line, whose channels are stored one plane after
    /// another, so locating a channel's samples needs the full `line_width`
    /// even when only `pixels.len()` of them starting at `x_start` are wanted.
    /// Passing `x_start = 0` and `line_width = pixels.len()` reads the
    /// complete line.
    fn read_pixels<FullPixel>(
        &self,
        bytes: &[u8],
        line_width: usize,
        x_start: usize,
        pixels: &mut [FullPixel],
        get_pixel: impl Fn(&mut FullPixel) -> &mut Self::RecursivePixel,
    );
}

// does not use the generic `Recursive` struct to reduce the number of angle
// brackets in the public api
/// Used to read another specific channel from an image.
/// Contains the previous `ReadChannels` objects.
#[derive(Clone, Debug)]
pub struct ReadOptionalChannel<ReadChannels, Sample> {
    previous_channels: ReadChannels,
    channel_name: Text,
    default_sample: Sample,
}

// does not use the generic `Recursive` struct to reduce the number of angle
// brackets in the public api
/// Used to read another specific channel from an image.
/// Contains the previous `ReadChannels` objects.
#[derive(Clone, Debug)]
pub struct ReadRequiredChannel<ReadChannels, Sample> {
    previous_channels: ReadChannels,
    channel_name: Text,
    px: PhantomData<Sample>,
}

/// Specifies how to collect all the specified channels into a number of
/// individual pixels.
#[derive(Copy, Clone, Debug)]
pub struct CollectPixels<ReadChannels, Pixel, PixelStorage, CreatePixels, SetPixel> {
    read_channels: ReadChannels,
    create_pixels: CreatePixels,
    set_pixel: SetPixel,
    px: PhantomData<(Pixel, PixelStorage)>,
}

/// Like `CollectPixels`, but for `collect_pixels_in_parallel`: pixels are
/// written via `set_row_pixel(row, x, pixel)` instead of `set_pixel(storage,
/// position, pixel)`, so worker threads only ever need access to their own
/// disjoint row range.
#[cfg(feature = "rayon")]
#[derive(Copy, Clone, Debug)]
pub struct CollectPixelsInParallel<ReadChannels, Pixel, PixelStorage, CreatePixels, SetRowPixel> {
    read_channels: ReadChannels,
    create_pixels: CreatePixels,
    set_row_pixel: SetRowPixel,
    px: PhantomData<(Pixel, PixelStorage)>,
}

impl<Inner: CheckDuplicates, Sample> CheckDuplicates for ReadRequiredChannel<Inner, Sample> {
    fn already_contains(&self, name: &Text) -> bool {
        &self.channel_name == name || self.previous_channels.already_contains(name)
    }
}

impl<Inner: CheckDuplicates, Sample> CheckDuplicates for ReadOptionalChannel<Inner, Sample> {
    fn already_contains(&self, name: &Text) -> bool {
        &self.channel_name == name || self.previous_channels.already_contains(name)
    }
}

impl<'s, InnerChannels, Pixel: 'static, PixelStorage: 'static, CreatePixels, SetPixel: 's>
ReadChannels<'s> for CollectPixels<InnerChannels, Pixel, PixelStorage, CreatePixels, SetPixel>
    where
        InnerChannels: ReadSpecificChannel,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
        CreatePixels: Fn(Vec2<usize>, &<<InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive) -> PixelStorage,
        SetPixel: Fn(&mut PixelStorage, Vec2<usize>, Pixel),
{
    type Reader = SpecificChannelsReader<
        PixelStorage, &'s SetPixel,
        InnerChannels::RecursivePixelReader,
        Pixel,
    >;

    fn create_channels_reader(&'s self, header: &Header) -> Result<Self::Reader> {
        if header.deep { return Err(Error::invalid("`SpecificChannels` does not support deep data yet")) }

        let pixel_reader = self.read_channels.create_recursive_reader(&header.channels)?;
        let channel_descriptions = pixel_reader.get_descriptions().into_non_recursive();// TODO not call this twice

        let create = &self.create_pixels;
        let pixel_storage = create(header.layer_size, &channel_descriptions);

        Ok(SpecificChannelsReader {
            set_pixel: &self.set_pixel,
            pixel_storage,
            pixel_reader,
            line_pixels: Vec::new(),
            px: Default::default()
        })
    }
}

#[cfg(feature = "rayon")]
impl<'s, InnerChannels, Pixel, PixelStorage, CreatePixels, SetRowPixel: 's>
ReadChannels<'s> for CollectPixelsInParallel<InnerChannels, Pixel, PixelStorage, CreatePixels, SetRowPixel>
    where
        InnerChannels: ReadSpecificChannel,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
        CreatePixels: Fn(Vec2<usize>, &<<InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive) -> PixelStorage,
        PixelStorage: RowMajorPixelStorage,
        PixelStorage::Element: Send,
        SetRowPixel: Fn(&mut [PixelStorage::Element], usize, Pixel) + Sync,
        InnerChannels::RecursivePixelReader: Sync,
        Pixel: Send,
{
    type Reader = SpecificChannelsParallelReader<
        PixelStorage, &'s SetRowPixel,
        InnerChannels::RecursivePixelReader,
        Pixel,
    >;

    fn create_channels_reader(&'s self, header: &Header) -> Result<Self::Reader> {
        if header.deep { return Err(Error::invalid("`SpecificChannels` does not support deep data yet")) }

        let pixel_reader = self.read_channels.create_recursive_reader(&header.channels)?;
        let channel_descriptions = pixel_reader.get_descriptions().into_non_recursive();

        let create = &self.create_pixels;
        let pixel_storage = create(header.layer_size, &channel_descriptions);

        Ok(SpecificChannelsParallelReader {
            set_row_pixel: &self.set_row_pixel,
            pixel_storage,
            pixel_reader,
            line_pixels: Vec::new(),
            px: Default::default()
        })
    }
}

/// The reader that holds the temporary data that is required to read some
/// specified channels.
#[derive(Clone, Debug)]
pub struct SpecificChannelsReader<PixelStorage, SetPixel, PixelReader, Pixel>
where
    PixelReader: RecursivePixelReader,
{
    set_pixel: SetPixel,
    pixel_storage: PixelStorage,
    pixel_reader: PixelReader,
    /// Converted-pixel scratch reused across blocks (grows to `LINE_RUN_PIXELS`).
    line_pixels: Vec<PixelReader::RecursivePixel>,
    px: PhantomData<Pixel>,
}

impl<PixelStorage: 'static, SetPixel, PxReader, Pixel: 'static> ChannelsReader
    for SpecificChannelsReader<PixelStorage, SetPixel, PxReader, Pixel>
where
    PxReader: RecursivePixelReader,
    PxReader::RecursivePixel: IntoTuple<Pixel>,
    PxReader::RecursiveChannelDescriptions: IntoNonRecursive,
    SetPixel: Fn(&mut PixelStorage, Vec2<usize>, Pixel),
{
    type Channels = SpecificChannels<
        PixelStorage,
        <PxReader::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive,
    >;

    fn filter_block(&self, tile: TileCoordinates) -> bool {
        tile.is_largest_resolution_level()
    }

    // TODO all levels

    fn read_block(&mut self, header: &Header, block: UncompressedBlock) -> UnitResult {
        let line_width = block.index.pixel_size.width();

        // The scratch line is walked once per channel by `read_pixels` and once
        // more by the write below. A whole 8K line of three halves is 48 KiB --
        // exactly one L1d -- so each of those passes evicts the one before it.
        // Working in runs that comfortably fit L1 keeps all of them hot; the
        // channel planes are addressed from `line_width`, so splitting the line
        // does not change which bytes are read.
        let run_width = LINE_RUN_PIXELS.min(line_width);
        if self.line_pixels.len() < run_width {
            self.line_pixels.resize(run_width, PxReader::RecursivePixel::default());
        }

        let byte_lines =
            block.data.chunks_exact(header.channels.bytes_per_pixel * line_width);
        debug_assert_eq!(
            byte_lines.len(),
            block.index.pixel_size.height(),
            "invalid block lines split"
        );

        let origin = block.index.pixel_position;
        for (y_offset, line_bytes) in byte_lines.enumerate() {
            // TODO sampling
            for x_start in (0..line_width).step_by(run_width) {
                let run = &mut self.line_pixels[..run_width.min(line_width - x_start)];

                // this two-step copy method should be very cache friendly in theory, and also
                // reduce sample_type lookup count
                self.pixel_reader.read_pixels(line_bytes, line_width, x_start, run, |px| px);

                // Prefer a row-hoisted write when `PixelStorage: RowMajorPixelStorage`
                // with `Element = Pixel`, otherwise call the opaque `set_pixel`
                // closure once per pixel
                write_decoded_line(
                    &mut self.pixel_storage,
                    &self.set_pixel,
                    origin.y() + y_offset,
                    origin.x() + x_start,
                    run.iter().map(|pixel| pixel.into_tuple()),
                );
            }
        }

        // every sample has been copied into the image, so the buffer can be
        // handed back for the next chunk to decompress into
        crate::block::pool::recycle(block.data);
        Ok(())
    }

    fn into_channels(self) -> Self::Channels {
        SpecificChannels {
            channels: self.pixel_reader.get_descriptions().into_non_recursive(),
            pixels: self.pixel_storage,
        }
    }
}

/// Like `SpecificChannelsReader`, but for `collect_pixels_in_parallel`: able
/// to write pixels directly from decompression worker threads, since
/// `PixelStorage: RowMajorPixelStorage` lets its rows be split into
/// disjoint, independently-writable ranges.
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct SpecificChannelsParallelReader<PixelStorage, SetRowPixel, PixelReader, Pixel>
where
    PixelReader: RecursivePixelReader,
{
    set_row_pixel: SetRowPixel,
    pixel_storage: PixelStorage,
    pixel_reader: PixelReader,
    /// Scratch for serial `read_block` / non-parallel fallback (workers keep their own).
    line_pixels: Vec<PixelReader::RecursivePixel>,
    px: PhantomData<Pixel>,
}

#[cfg(feature = "rayon")]
impl<PixelStorage, SetRowPixel, PxReader, Pixel> ChannelsReader
    for SpecificChannelsParallelReader<PixelStorage, SetRowPixel, PxReader, Pixel>
where
    PxReader: RecursivePixelReader + Sync,
    PxReader::RecursivePixel: IntoTuple<Pixel>,
    PxReader::RecursiveChannelDescriptions: IntoNonRecursive,
    PixelStorage: RowMajorPixelStorage,
    PixelStorage::Element: Send,
    SetRowPixel: Fn(&mut [PixelStorage::Element], usize, Pixel) + Sync,
    Pixel: Send,
{
    type Channels = SpecificChannels<
        PixelStorage,
        <PxReader::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive,
    >;

    fn filter_block(&self, tile: TileCoordinates) -> bool {
        tile.is_largest_resolution_level()
    }

    // Serial fallback, used whenever the caller isn't going through
    // `read_blocks_in_parallel` (e.g. `.non_parallel()` reads). Identical in
    // spirit to `SpecificChannelsReader::read_block`, just addressing
    // `pixel_storage` through `RowMajorPixelStorage::pixels_mut()` (a flat,
    // row-major buffer) and calling `set_row_pixel(row, x, pixel)` instead of
    // `set_pixel(storage, position, pixel)`.
    fn read_block(&mut self, header: &Header, block: UncompressedBlock) -> UnitResult {
        let line_width = block.index.pixel_size.width();
        if self.line_pixels.len() < line_width {
            self.line_pixels.resize(line_width, PxReader::RecursivePixel::default());
        }
        let pixels = &mut self.line_pixels[..line_width];

        let byte_lines = block
            .data
            .chunks_exact(header.channels.bytes_per_pixel * line_width);
        debug_assert_eq!(
            byte_lines.len(),
            block.index.pixel_size.height(),
            "invalid block lines split"
        );

        let storage_width = self.pixel_storage.width();
        let flat = self.pixel_storage.pixels_mut();
        let x0 = block.index.pixel_position.x();

        for (y_offset, line_bytes) in byte_lines.enumerate() {
            self.pixel_reader.read_pixels(line_bytes, line_width, 0, pixels, |px| px);
            let y = block.index.pixel_position.y() + y_offset;
            let row = &mut flat[y * storage_width .. (y + 1) * storage_width];

            for (x_offset, pixel) in pixels.iter().enumerate() {
                let set_row_pixel = &self.set_row_pixel;
                set_row_pixel(row, x0 + x_offset, pixel.into_tuple());
            }
        }

        // see the comment on the equivalent line in `SpecificChannelsReader`
        crate::block::pool::recycle(block.data);
        Ok(())
    }

    fn into_channels(self) -> Self::Channels {
        SpecificChannels {
            channels: self.pixel_reader.get_descriptions().into_non_recursive(),
            pixels: self.pixel_storage,
        }
    }

    fn supports_parallel_write(&self) -> bool {
        true
    }

    // Decompress+convert on workers as chunks arrive (read overlaps decompress).
    // Scanline: exclusive row bands via split_at_mut. Tiled: one mutex per row
    // (tiles share y; no unsafe strip split) with block-index x/width. Decreasing
    // line order: serial fallback (rare; avoid buffering all chunks to re-sort).
    fn read_blocks_in_parallel<R: crate::block::reader::ChunksReader + Send>(
        &mut self,
        header: &Header,
        mut chunks: R,
        meta_data: &crate::meta::MetaData,
        pool: &rayon_core::ThreadPool,
        pedantic: bool,
    ) -> UnitResult {
        let bytes_per_pixel = header.channels.bytes_per_pixel;

        if header.line_order == crate::meta::attribute::LineOrder::Decreasing {
            while let Some(chunk) = chunks.next() {
                let block = UncompressedBlock::decompress_chunk(chunk?, meta_data, pedantic)?;
                self.read_block(header, block)?;
            }
            return Ok(());
        }

        let is_tiled = !matches!(header.blocks, crate::meta::BlockDescription::ScanLines);
        if is_tiled {
            return self.read_tiled_blocks_in_parallel(header, chunks, meta_data, pool, pedantic);
        }

        let width = header.layer_size.width();
        let pixel_reader = &self.pixel_reader;
        let set_row_pixel = &self.set_row_pixel;
        let error: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);

        // A flat, contiguous buffer (rather than one allocation per row) so
        // each worker's row range is obtained by splitting this slice
        // directly -- no per-row indirection, and no threads concurrently
        // touching scattered, independently heap-allocated rows.
        let storage_width = self.pixel_storage.width();
        let mut flat: &mut [PixelStorage::Element] = self.pixel_storage.pixels_mut();
        let total_rows = if storage_width == 0 { 0 } else { flat.len() / storage_width };
        let mut cursor = 0usize;

        // Scheduling instrumentation (see `dwa::profile`'s SCHED_* counters).
        // Note `ThreadPool::scope` runs the feeding closure *on a pool worker*,
        // not on the calling thread, so the feeder occupies one of the N threads
        // for the whole span -- which is why `main_feed` is reported next to
        // worker busy time rather than treated as free.
        #[cfg(feature = "dwa-profile")]
        let scope_start = std::time::Instant::now();
        #[cfg(feature = "dwa-profile")]
        let mut feed_ns = 0u64;
        #[cfg(feature = "dwa-profile")]
        let feed_ns = &mut feed_ns;

        pool.scope(|scope| {
            #[cfg(feature = "dwa-profile")]
            let mut feed_mark = std::time::Instant::now();

            while let Some(chunk) = chunks.next() {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(new_error) => { *error.lock().unwrap() = Some(new_error); break; }
                };

                let (y, height) = match header
                    .get_block_data_indices(&chunk.compressed_block)
                    .and_then(|tile| header.get_absolute_block_pixel_coordinates(tile))
                {
                    Ok(indices) => (indices.position.y() as usize, indices.size.height()),
                    Err(new_error) => { *error.lock().unwrap() = Some(new_error); break; }
                };

                if y < cursor || y - cursor > total_rows || height > total_rows - (y - cursor) {
                    *error.lock().unwrap() = Some(Error::invalid("chunk row range"));
                    break;
                }

                let (_, rest) = flat.split_at_mut((y - cursor) * storage_width);
                let (this_elements, rest) = rest.split_at_mut(height * storage_width);
                flat = rest;
                cursor = y + height;

                let error = &error;

                #[cfg(feature = "dwa-profile")]
                let spawned_at = std::time::Instant::now();

                scope.spawn(move |_| {
                    #[cfg(feature = "dwa-profile")]
                    let task_start = {
                        use crate::compression::dwa::profile;
                        use std::sync::atomic::Ordering;
                        let now = std::time::Instant::now();
                        profile::SCHED_QUEUE_NS.fetch_add(
                            now.duration_since(spawned_at).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        profile::SCHED_TASKS.fetch_add(1, Ordering::Relaxed);
                        now
                    };

                    let result = UncompressedBlock::decompress_chunk(chunk, meta_data, pedantic)
                        .and_then(|block| {
                            let mut pixels = vec![PxReader::RecursivePixel::default(); width];

                            // Scanline bands are always full image width starting at x=0.
                            // (Tried the same L1-resident run tiling as the serial path
                            // here; at 8 threads workers are bandwidth-bound so the extra
                            // loop is pure overhead)
                            for (y_offset, line_bytes) in
                                block.data.chunks_exact(bytes_per_pixel * width).enumerate()
                            {
                                pixel_reader.read_pixels(line_bytes, width, 0, &mut pixels, |px| px);
                                let row = &mut this_elements
                                    [y_offset * storage_width .. (y_offset + 1) * storage_width];

                                for (x_offset, pixel) in pixels.iter().enumerate() {
                                    set_row_pixel(row, x_offset, pixel.into_tuple());
                                }
                            }

                            // recycled from the same worker that decompressed
                            // into it, so the next chunk on this thread can
                            // reuse the pages it just faulted in
                            crate::block::pool::recycle(block.data);
                            Ok(())
                        });

                    if let Err(new_error) = result {
                        *error.lock().unwrap() = Some(new_error);
                    }

                    #[cfg(feature = "dwa-profile")]
                    crate::compression::dwa::profile::SCHED_TASK_NS.fetch_add(
                        task_start.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                });

                #[cfg(feature = "dwa-profile")]
                {
                    let now = std::time::Instant::now();
                    *feed_ns += now.duration_since(feed_mark).as_nanos() as u64;
                    feed_mark = now;
                }
            }
        });

        // Everything inside the scope that wasn't the feeding loop is the
        // feeder parked on the scope latch waiting for stragglers -- the
        // tail-imbalance cost of the last wave of chunks.
        #[cfg(feature = "dwa-profile")]
        {
            use crate::compression::dwa::profile;
            use std::sync::atomic::Ordering;
            let wall = scope_start.elapsed().as_nanos() as u64;
            profile::SCHED_WALL_NS.fetch_add(wall, Ordering::Relaxed);
            profile::SCHED_FEED_NS.fetch_add(*feed_ns, Ordering::Relaxed);
            profile::SCHED_DRAIN_NS.fetch_add(wall.saturating_sub(*feed_ns), Ordering::Relaxed);
            profile::SCHED_THREADS.store(pool.current_num_threads() as u64, Ordering::Relaxed);
        }

        match error.into_inner().unwrap() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(feature = "rayon")]
impl<PixelStorage, SetRowPixel, PxReader, Pixel>
    SpecificChannelsParallelReader<PixelStorage, SetRowPixel, PxReader, Pixel>
where
    PxReader: RecursivePixelReader + Sync,
    PxReader::RecursivePixel: IntoTuple<Pixel>,
    PxReader::RecursiveChannelDescriptions: IntoNonRecursive,
    PixelStorage: RowMajorPixelStorage,
    PixelStorage::Element: Send,
    SetRowPixel: Fn(&mut [PixelStorage::Element], usize, Pixel) + Sync,
    Pixel: Send,
{
    /// Parallel path for tiled files: decompress + convert on workers, write
    /// under a per-row mutex. See `read_blocks_in_parallel`.
    fn read_tiled_blocks_in_parallel<R: crate::block::reader::ChunksReader + Send>(
        &mut self,
        header: &Header,
        mut chunks: R,
        meta_data: &crate::meta::MetaData,
        pool: &rayon_core::ThreadPool,
        pedantic: bool,
    ) -> UnitResult {
        let bytes_per_pixel = header.channels.bytes_per_pixel;
        let pixel_reader = &self.pixel_reader;
        let set_row_pixel = &self.set_row_pixel;
        let error: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);

        let storage_width = self.pixel_storage.width();
        let flat = self.pixel_storage.pixels_mut();
        let total_rows = if storage_width == 0 { 0 } else { flat.len() / storage_width };
        // One mutex per row, not one mutex for the whole image. Tiles only
        // ever contend when two of them share a row (same tile-row band,
        // different tile-columns); `chunks_exact_mut` (safe)` to allow splitting a row-major buffer into
        // concurrent vertical tile strips) hands out non-overlapping row
        // slices up front, so unrelated tile-rows never touch the same lock.
        let rows: Vec<std::sync::Mutex<&mut [PixelStorage::Element]>> = if storage_width == 0 {
            Vec::new()
        } else {
            flat.chunks_exact_mut(storage_width).map(std::sync::Mutex::new).collect()
        };

        pool.scope(|scope| {
            while let Some(chunk) = chunks.next() {
                let chunk = match chunk {
                    Ok(chunk) => chunk,
                    Err(new_error) => {
                        *error.lock().unwrap() = Some(new_error);
                        break;
                    }
                };

                let (x0, y0, block_width, block_height) = match header
                    .get_block_data_indices(&chunk.compressed_block)
                    .and_then(|tile| header.get_absolute_block_pixel_coordinates(tile))
                {
                    Ok(indices) => {
                        if indices.position.x() < 0 || indices.position.y() < 0 {
                            *error.lock().unwrap() = Some(Error::invalid("chunk pixel position"));
                            break;
                        }
                        (
                            indices.position.x() as usize,
                            indices.position.y() as usize,
                            indices.size.width(),
                            indices.size.height(),
                        )
                    }
                    Err(new_error) => {
                        *error.lock().unwrap() = Some(new_error);
                        break;
                    }
                };

                if y0 > total_rows
                    || block_height > total_rows - y0
                    || x0 > storage_width
                    || block_width > storage_width - x0
                {
                    *error.lock().unwrap() = Some(Error::invalid("chunk tile range"));
                    break;
                }

                let error = &error;
                let rows = &rows;
                scope.spawn(move |_| {
                    let result = UncompressedBlock::decompress_chunk(chunk, meta_data, pedantic)
                        .and_then(|block| {
                            debug_assert_eq!(block.index.pixel_size.width(), block_width);
                            debug_assert_eq!(block.index.pixel_size.height(), block_height);
                            debug_assert_eq!(block.index.pixel_position.x(), x0);
                            debug_assert_eq!(block.index.pixel_position.y(), y0);

                            let mut pixels =
                                vec![PxReader::RecursivePixel::default(); block_width];

                            for (y_offset, line_bytes) in block
                                .data
                                .chunks_exact(bytes_per_pixel * block_width)
                                .enumerate()
                            {
                                // Sample conversion stays outside the lock;
                                // only this row's write is serialized, and
                                // only against other tiles that share this
                                // exact row.
                                pixel_reader.read_pixels(
                                    line_bytes,
                                    block_width,
                                    0,
                                    &mut pixels,
                                    |px| px,
                                );

                                let y = y0 + y_offset;
                                let mut row = rows[y].lock().unwrap();
                                for (x_offset, pixel) in pixels.iter().enumerate() {
                                    set_row_pixel(&mut row, x0 + x_offset, pixel.into_tuple());
                                }
                            }

                            crate::block::pool::recycle(block.data);
                            Ok(())
                        });

                    if let Err(new_error) = result {
                        *error.lock().unwrap() = Some(new_error);
                    }
                });
            }
        });

        match error.into_inner().unwrap() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// Read zero channels from an image. Call `with_named_channel` on this object
/// to read as many channels as desired.
pub type ReadZeroChannels = NoneMore;

impl ReadSpecificChannel for NoneMore {
    type RecursivePixelReader = Self;

    fn create_recursive_reader(&self, _: &ChannelList) -> Result<Self::RecursivePixelReader> {
        Ok(Self)
    }
}

impl<DefaultSample, ReadChannels> ReadSpecificChannel
    for ReadOptionalChannel<ReadChannels, DefaultSample>
where
    ReadChannels: ReadSpecificChannel,
    DefaultSample: FromNativeSample + 'static,
{
    type RecursivePixelReader =
        Recursive<ReadChannels::RecursivePixelReader, OptionalSampleReader<DefaultSample>>;

    fn create_recursive_reader(
        &self,
        channels: &ChannelList,
    ) -> Result<Self::RecursivePixelReader> {
        debug_assert!(
            self.previous_channels.already_contains(&self.channel_name).not(),
            "duplicate channel name: {}",
            self.channel_name
        );

        let inner_samples_reader = self.previous_channels.create_recursive_reader(channels)?;
        let reader = channels
            .channels_with_byte_offset()
            .find(|(_, channel)| channel.name == self.channel_name)
            .map(|(channel_byte_offset, channel)| SampleReader {
                channel_byte_offset,
                channel: channel.clone(),
                px: Default::default(),
            });

        Ok(Recursive::new(
            inner_samples_reader,
            OptionalSampleReader {
                reader,
                default_sample: self.default_sample,
            },
        ))
    }
}

impl<Sample, ReadChannels> ReadSpecificChannel for ReadRequiredChannel<ReadChannels, Sample>
where
    ReadChannels: ReadSpecificChannel,
    Sample: FromNativeSample + 'static,
{
    type RecursivePixelReader = Recursive<ReadChannels::RecursivePixelReader, SampleReader<Sample>>;

    fn create_recursive_reader(
        &self,
        channels: &ChannelList,
    ) -> Result<Self::RecursivePixelReader> {
        let previous_samples_reader = self.previous_channels.create_recursive_reader(channels)?;
        let (channel_byte_offset, channel) = channels
            .channels_with_byte_offset()
            .find(|(_, channel)| channel.name == self.channel_name)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "layer does not contain all of your specified channels (`{}` is missing)",
                    self.channel_name
                ))
            })?;

        Ok(Recursive::new(
            previous_samples_reader,
            SampleReader {
                channel_byte_offset,
                channel: channel.clone(),
                px: Default::default(),
            },
        ))
    }
}

/// Reader for a single channel. Generic over the concrete sample type (f16,
/// f32, u32).
#[derive(Clone, Debug)]
pub struct SampleReader<Sample> {
    /// to be multiplied with line width!
    channel_byte_offset: usize,

    channel: ChannelDescription,
    px: PhantomData<Sample>,
}

/// Reader for a single channel. Generic over the concrete sample type (f16,
/// f32, u32). Can also skip reading a channel if it could not be found in the
/// image.
#[derive(Clone, Debug)]
pub struct OptionalSampleReader<DefaultSample> {
    reader: Option<SampleReader<DefaultSample>>,
    default_sample: DefaultSample,
}

impl<Sample: FromNativeSample> SampleReader<Sample> {
    fn read_own_samples<FullPixel>(
        &self,
        bytes: &[u8],
        line_width: usize,
        x_start: usize,
        pixels: &mut [FullPixel],
        get_sample: impl Fn(&mut FullPixel) -> &mut Sample,
    ) {
        // this channel's plane starts after the planes of all preceding
        // channels, which are `line_width` samples wide, so the plane base
        // scales with the whole line, while the run inside it scales with the
        // requested pixel count
        let bytes_per_sample = self.channel.sample_type.bytes_per_sample();
        let start_index = line_width * self.channel_byte_offset + x_start * bytes_per_sample;
        let byte_count = pixels.len() * bytes_per_sample;
        let mut own_bytes_reader = &mut &bytes[start_index..start_index + byte_count]; // TODO check block size somewhere
        let mut samples_out = pixels.iter_mut().map(get_sample);

        // match the type once for the whole line, not on every single sample
        match self.channel.sample_type {
            SampleType::F16 => read_and_convert_all_samples_batched(
                &mut own_bytes_reader,
                &mut samples_out,
                Sample::from_f16s,
            ),

            SampleType::F32 => read_and_convert_all_samples_batched(
                &mut own_bytes_reader,
                &mut samples_out,
                Sample::from_f32s,
            ),

            SampleType::U32 => read_and_convert_all_samples_batched(
                &mut own_bytes_reader,
                &mut samples_out,
                Sample::from_u32s,
            ),
        }

        debug_assert!(samples_out.next().is_none(), "not all samples have been converted");
        debug_assert!(own_bytes_reader.is_empty(), "bytes left after reading all samples");
    }
}

/// Does the same as `convert_batch(in_bytes.chunks().map(From::from_bytes))`,
/// but vectorized. Reads the samples for one line, using the sample type
/// specified in the file, and then converts those to the desired sample types.
/// Uses batches to allow vectorization, converting multiple values with one
/// instruction. Does not convert endianness.
fn read_and_convert_all_samples_batched<'t, From, To>(
    mut in_bytes: impl Read,
    out_samples: &mut impl ExactSizeIterator<Item = &'t mut To>,
    convert_batch: fn(&[From], &mut [To]),
) where
    From: Data + Default + Copy,
    To: 't + Default + Copy,
{
    // this is not a global! why is this warning triggered?
    #[allow(non_upper_case_globals)]
    const batch_size: usize = 16;

    let total_sample_count = out_samples.len();
    let batch_count = total_sample_count / batch_size;
    let remaining_samples_count = total_sample_count % batch_size;

    let len_error_msg = "sample count was miscalculated";
    let byte_error_msg = "error when reading from in-memory slice";

    // write samples from a given slice to the output iterator. should be inlined.
    let output_n_samples = &mut move |samples: &[To]| {
        for converted_sample in samples {
            *out_samples.next().expect(len_error_msg) = *converted_sample;
        }
    };

    // read samples from the byte source into a given slice. should be inlined.
    // todo: use #[inline] when available
    // error[E0658]: attributes on expressions are experimental,
    // see issue #15701 <https://github.com/rust-lang/rust/issues/15701> for more information
    let read_n_samples = &mut move |samples: &mut [From]| {
        Data::read_slice_ne(&mut in_bytes, samples).expect(byte_error_msg);
    };

    // temporary arrays with fixed size, operations should be vectorized within
    // these arrays
    let mut source_samples_batch: [From; batch_size] = Default::default();
    let mut desired_samples_batch: [To; batch_size] = Default::default();

    // first convert all whole batches, size statically known to be 16 element
    // arrays
    for _ in 0..batch_count {
        read_n_samples(&mut source_samples_batch);
        convert_batch(source_samples_batch.as_slice(), desired_samples_batch.as_mut_slice());
        output_n_samples(&desired_samples_batch);
    }

    // then convert a partial remaining batch, size known only at runtime
    if remaining_samples_count != 0 {
        let source_samples_batch = &mut source_samples_batch[..remaining_samples_count];
        let desired_samples_batch = &mut desired_samples_batch[..remaining_samples_count];

        read_n_samples(source_samples_batch);
        convert_batch(source_samples_batch, desired_samples_batch);
        output_n_samples(desired_samples_batch);
    }
}

#[cfg(test)]
mod test {
    use half::f16;

    use super::*;

    #[test]
    fn equals_naive_f32() {
        for total_array_size in [3, 7, 30, 41, 120, 10_423] {
            let input_f32s =
                (0..total_array_size).map(|_| rand::random::<f32>()).collect::<Vec<f32>>();
            let in_f32s_bytes =
                input_f32s.iter().cloned().flat_map(f32::to_ne_bytes).collect::<Vec<u8>>();

            let mut out_f16_samples_batched =
                vec![f16::from_f32(rand::random::<f32>()); total_array_size];

            read_and_convert_all_samples_batched(
                &mut in_f32s_bytes.as_slice(),
                &mut out_f16_samples_batched.iter_mut(),
                f16::from_f32s,
            );

            let out_f16_samples_naive = input_f32s.iter().cloned().map(f16::from_f32);

            assert!(out_f16_samples_naive.eq(out_f16_samples_batched));
        }
    }

    /// Flat (`RowMajorPixelStorage<Element = Pixel>`) and nested (`Vec<Vec<_>>`)
    /// write paths must produce the same layout for the same decoded line.
    #[test]
    fn write_decoded_line_flat_matches_nested() {
        let width = 8usize;
        let height = 3usize;
        let line: Vec<(f32, f32, f32)> =
            (0..width).map(|x| (x as f32, x as f32 * 2.0, x as f32 * 3.0)).collect();

        let mut flat = FlatRowMajorPixelStorage {
            width,
            pixels: vec![(0.0f32, 0.0, 0.0); width * height],
        };
        let mut nested = vec![vec![(0.0f32, 0.0, 0.0); width]; height];

        for y in 0..height {
            {
                let set = |_: &mut FlatRowMajorPixelStorage<(f32, f32, f32)>,
                           _: Vec2<usize>,
                           _: (f32, f32, f32)| {
                    panic!("flat path must not call set_pixel when Element = Pixel");
                };
                write_decoded_line(&mut flat, &set, y, 0, line.iter().copied());
            }
            {
                let set = |img: &mut Vec<Vec<(f32, f32, f32)>>, pos: Vec2<usize>, px| {
                    img[pos.y()][pos.x()] = px;
                };
                write_decoded_line(&mut nested, &set, y, 0, line.iter().copied());
            }
        }

        for y in 0..height {
            assert_eq!(&flat.pixels[y * width..(y + 1) * width], nested[y].as_slice());
        }
    }
}

impl RecursivePixelReader for NoneMore {
    type RecursiveChannelDescriptions = Self;
    type RecursivePixel = Self;

    fn get_descriptions(&self) -> Self::RecursiveChannelDescriptions {
        Self
    }

    fn read_pixels<FullPixel>(
        &self,
        _: &[u8],
        _: usize,
        _: usize,
        _: &mut [FullPixel],
        _: impl Fn(&mut FullPixel) -> &mut Self,
    ) {
    }
}

impl<Sample, InnerReader: RecursivePixelReader> RecursivePixelReader
    for Recursive<InnerReader, SampleReader<Sample>>
where
    Sample: FromNativeSample + 'static,
{
    type RecursiveChannelDescriptions =
        Recursive<InnerReader::RecursiveChannelDescriptions, ChannelDescription>;
    type RecursivePixel = Recursive<InnerReader::RecursivePixel, Sample>;

    fn get_descriptions(&self) -> Self::RecursiveChannelDescriptions {
        Recursive::new(self.inner.get_descriptions(), self.value.channel.clone())
    }

    fn read_pixels<FullPixel>(
        &self,
        bytes: &[u8],
        line_width: usize,
        x_start: usize,
        pixels: &mut [FullPixel],
        get_pixel: impl Fn(&mut FullPixel) -> &mut Self::RecursivePixel,
    ) {
        self.value.read_own_samples(bytes, line_width, x_start, pixels, |px| {
            &mut get_pixel(px).value
        });
        self.inner.read_pixels(bytes, line_width, x_start, pixels, |px| {
            &mut get_pixel(px).inner
        });
    }
}

impl<Sample, InnerReader: RecursivePixelReader> RecursivePixelReader
    for Recursive<InnerReader, OptionalSampleReader<Sample>>
where
    Sample: FromNativeSample + 'static,
{
    type RecursiveChannelDescriptions =
        Recursive<InnerReader::RecursiveChannelDescriptions, Option<ChannelDescription>>;
    type RecursivePixel = Recursive<InnerReader::RecursivePixel, Sample>;

    fn get_descriptions(&self) -> Self::RecursiveChannelDescriptions {
        Recursive::new(
            self.inner.get_descriptions(),
            self.value.reader.as_ref().map(|reader| reader.channel.clone()),
        )
    }

    fn read_pixels<FullPixel>(
        &self,
        bytes: &[u8],
        line_width: usize,
        x_start: usize,
        pixels: &mut [FullPixel],
        get_pixel: impl Fn(&mut FullPixel) -> &mut Self::RecursivePixel,
    ) {
        if let Some(reader) = &self.value.reader {
            reader.read_own_samples(bytes, line_width, x_start, pixels, |px| {
                &mut get_pixel(px).value
            });
        } else {
            // if this channel is optional and was not found in the file, fill the default
            // sample
            for pixel in pixels.iter_mut() {
                get_pixel(pixel).value = self.value.default_sample;
            }
        }

        self.inner.read_pixels(bytes, line_width, x_start, pixels, |px| {
            &mut get_pixel(px).inner
        });
    }
}
