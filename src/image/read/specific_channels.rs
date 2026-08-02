//! How to read arbitrary but specific selection of arbitrary channels.
//! This is not a zero-cost abstraction.

use std::any::Any;
use std::cell::RefCell;
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
/// be read from the file. Then finish with:
/// - **[`collect_flat_pixels`](Self::collect_flat_pixels)** (preferred); one
///   contiguous buffer + [`PixelSink`]; parallel decompress+write with `rayon`
/// - [`collect_pixels`](Self::collect_pixels)
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

    /// Define how to store pixels for custom / non-flat storage.
    ///
    /// The first closure creates the storage; the second inserts one pixel at
    /// a `Vec2` position. Pixel samples must be `f16`, `f32`, `u32`, or
    /// `Sample`
    ///
    /// Prefer [`collect_flat_pixels`](Self::collect_flat_pixels) when using
    /// a single contiguous row-major buffer ([`FlatRowMajorPixelStorage`]
    /// or [`pixel_vec::PixelVec`]) -> path is the performance default and
    /// can write from decompression workers
    ///
    /// When `PixelStorage` implements [`RowMajorPixelStorage`] with
    /// `Element = Pixel`, the serial reader may write into hoisted row slices
    /// and not call `set_pixel`. Use a non-identity `set_pixel` only with
    /// non-row-major storage, or transform after the read
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

    /// Preferred path: flat row-major storage + one [`PixelSink`].
    #[cfg(feature = "rayon")]
    fn collect_flat_pixels<Pixel, PixelStorage, CreatePixels, Sink>(
        self, create_pixels: CreatePixels, sink: Sink
    ) -> CollectFlatPixels<Self, Pixel, PixelStorage, CreatePixels, Sink>
        where
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
            CreatePixels: Fn(
                Vec2<usize>,
                &<<Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive
            ) -> PixelStorage,
            PixelStorage: RowMajorPixelStorage,
            // `Clone` so each channels-reader can own a sink (parallel workers share via `Sync`).
            Sink: PixelSink<PixelStorage::Element, Pixel> + Clone,
    {
        CollectFlatPixels {
            read_channels: self,
            sink,
            create_pixels,
            px: Default::default(),
        }
    }

    /// [`collect_flat_pixels`](Self::collect_flat_pixels) with [`CopyPixel`]
    #[cfg(feature = "rayon")]
    fn collect_flat_pixels_copy<Pixel, PixelStorage, CreatePixels>(
        self, create_pixels: CreatePixels
    ) -> CollectFlatPixels<Self, Pixel, PixelStorage, CreatePixels, CopyPixel>
        where
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
            CreatePixels: Fn(
                Vec2<usize>,
                &<<Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive
            ) -> PixelStorage,
            PixelStorage: RowMajorPixelStorage<Element = Pixel>,
            Pixel: Copy,
    {
        self.collect_flat_pixels(create_pixels, CopyPixel)
    }

    /// Alias for [`collect_flat_pixels`](Self::collect_flat_pixels).
    #[cfg(feature = "rayon")]
    fn collect_pixels_in_parallel<Pixel, PixelStorage, CreatePixels, Sink>(
        self, create_pixels: CreatePixels, set_row_pixel: Sink
    ) -> CollectFlatPixels<Self, Pixel, PixelStorage, CreatePixels, Sink>
        where
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
            <Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
            CreatePixels: Fn(
                Vec2<usize>,
                &<<Self::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive
            ) -> PixelStorage,
            PixelStorage: RowMajorPixelStorage,
            Sink: PixelSink<PixelStorage::Element, Pixel> + Clone,
    {
        self.collect_flat_pixels(create_pixels, set_row_pixel)
    }
}

/// Parallel-safe write of one decoded pixel into a flat row.
pub trait PixelSink<Element, Pixel> {
    /// Write `pixel` at column `x` of `row`.
    fn write(&self, row: &mut [Element], x: usize, pixel: Pixel);
}

impl<F, Element, Pixel> PixelSink<Element, Pixel> for F
where
    F: Fn(&mut [Element], usize, Pixel),
{
    #[inline]
    fn write(&self, row: &mut [Element], x: usize, pixel: Pixel) {
        (self)(row, x, pixel);
    }
}

/// [`PixelSink`] that copies the decoded pixel into the row (`Element == Pixel`).
#[derive(Copy, Clone, Debug, Default)]
pub struct CopyPixel;

impl<P: Copy> PixelSink<P, P> for CopyPixel {
    #[inline]
    fn write(&self, row: &mut [P], x: usize, pixel: P) {
        row[x] = pixel;
    }
}

/// Pixel storage backed by one contiguous, row-major buffer.
///
/// Required by [`collect_flat_pixels`](ReadSpecificChannel::collect_flat_pixels)
/// (and its alias `collect_pixels_in_parallel`)
///
/// Contiguity matters for parallel decompress
pub trait RowMajorPixelStorage {
    /// Per-slot type in the flat buffer. The [`PixelSink`] may convert the
    /// decoded `Pixel` into this type (or copy it when they match — see
    /// [`CopyPixel`]).
    ///
    /// When `Element` equals the decoded `Pixel` in
    /// [`collect_pixels`](ReadSpecificChannel::collect_pixels), the serial
    /// reader may write row slices without calling `set_pixel`.
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

#[inline]
fn line_run_pixels<Pixel>(channel_count: usize) -> usize {
    let element_bytes = std::mem::size_of::<Pixel>().max(1);
    let passes = channel_count.max(1).saturating_add(1); // per-channel read + write
    crate::cpu_cache::l1_resident_count(element_bytes, passes)
}

// TLS `Vec<T>` via `Any` monomorphize pixel/sample types share one slot
fn take_tls_any_vec<P: Default + Clone + 'static>(
    cell: &RefCell<Option<Box<dyn Any>>>,
    len: usize,
) -> Vec<P> {
    let mut slot = cell.borrow_mut();
    let mut vec = match slot.take() {
        Some(boxed) => match boxed.downcast::<Vec<P>>() {
            Ok(v) => *v,
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    };
    if vec.len() < len {
        vec.resize(len, P::default());
    }
    vec
}

fn return_tls_any_vec<P: 'static>(cell: &RefCell<Option<Box<dyn Any>>>, vec: Vec<P>) {
    *cell.borrow_mut() = Some(Box::new(vec));
}

// Per-worker convert-line scratch for parallel readers (`RecursivePixel`).
#[cfg(feature = "rayon")]
thread_local! {
    static WORKER_LINE_SCRATCH: RefCell<Option<Box<dyn Any>>> =
        const { RefCell::new(None) };
}

#[cfg(feature = "rayon")]
#[inline]
fn take_worker_line_scratch<P: Default + Clone + 'static>(width: usize) -> Vec<P> {
    WORKER_LINE_SCRATCH.with(|cell| take_tls_any_vec(cell, width))
}

#[cfg(feature = "rayon")]
#[inline]
fn return_worker_line_scratch<P: 'static>(vec: Vec<P>) {
    WORKER_LINE_SCRATCH.with(|cell| return_tls_any_vec(cell, vec));
}

// Channel-plane convert scratch for `read_and_convert_all_samples_batched`
thread_local! {
    static CONVERT_FROM_SCRATCH: RefCell<Option<Box<dyn Any>>> =
        const { RefCell::new(None) };
    static CONVERT_TO_SCRATCH: RefCell<Option<Box<dyn Any>>> =
        const { RefCell::new(None) };
}

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

/// Flat-buffer collect config: [`RowMajorPixelStorage`] + [`PixelSink`].
///
/// Produced by [`ReadSpecificChannel::collect_flat_pixels`] (and the
/// `collect_pixels_in_parallel` alias). Workers only see row slices.
#[cfg(feature = "rayon")]
#[derive(Copy, Clone, Debug)]
pub struct CollectFlatPixels<ReadChannels, Pixel, PixelStorage, CreatePixels, Sink> {
    read_channels: ReadChannels,
    create_pixels: CreatePixels,
    sink: Sink,
    px: PhantomData<(Pixel, PixelStorage)>,
}

/// Old name for [`CollectFlatPixels`].
#[cfg(feature = "rayon")]
pub type CollectPixelsInParallel<ReadChannels, Pixel, PixelStorage, CreatePixels, Sink> =
    CollectFlatPixels<ReadChannels, Pixel, PixelStorage, CreatePixels, Sink>;

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
impl<'s, InnerChannels, Pixel, PixelStorage, CreatePixels, Sink>
ReadChannels<'s> for CollectFlatPixels<InnerChannels, Pixel, PixelStorage, CreatePixels, Sink>
    where
        InnerChannels: ReadSpecificChannel,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursivePixel: IntoTuple<Pixel>,
        <InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions: IntoNonRecursive,
        CreatePixels: Fn(Vec2<usize>, &<<InnerChannels::RecursivePixelReader as RecursivePixelReader>::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive) -> PixelStorage,
        PixelStorage: RowMajorPixelStorage,
        PixelStorage::Element: Send,
        Sink: PixelSink<PixelStorage::Element, Pixel> + Sync + Clone + 's,
        InnerChannels::RecursivePixelReader: Sync,
        Pixel: Send,
{
    type Reader = SpecificChannelsParallelReader<
        PixelStorage, Sink,
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
            sink: self.sink.clone(),
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
    /// Per-block scratch (grows to L1 run width).
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

        let run_width = line_run_pixels::<PxReader::RecursivePixel>(header.channels.list.len())
            .min(line_width);
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

/// Flat-storage reader for [`collect_flat_pixels`](ReadSpecificChannel::collect_flat_pixels):
/// workers write via [`PixelSink`] into row slices of a [`RowMajorPixelStorage`]
#[cfg(feature = "rayon")]
#[derive(Clone, Debug)]
pub struct SpecificChannelsParallelReader<PixelStorage, Sink, PixelReader, Pixel>
where
    PixelReader: RecursivePixelReader,
{
    sink: Sink,
    pixel_storage: PixelStorage,
    pixel_reader: PixelReader,
    /// Scratch for serial `read_block` / non-parallel fallback (workers keep their own).
    line_pixels: Vec<PixelReader::RecursivePixel>,
    px: PhantomData<Pixel>,
}

#[cfg(feature = "rayon")]
impl<PixelStorage, Sink, PxReader, Pixel> ChannelsReader
    for SpecificChannelsParallelReader<PixelStorage, Sink, PxReader, Pixel>
where
    PxReader: RecursivePixelReader + Sync,
    PxReader::RecursivePixel: IntoTuple<Pixel>,
    PxReader::RecursiveChannelDescriptions: IntoNonRecursive,
    PixelStorage: RowMajorPixelStorage,
    PixelStorage::Element: Send,
    Sink: PixelSink<PixelStorage::Element, Pixel> + Sync,
    Pixel: Send,
{
    type Channels = SpecificChannels<
        PixelStorage,
        <PxReader::RecursiveChannelDescriptions as IntoNonRecursive>::NonRecursive,
    >;

    fn filter_block(&self, tile: TileCoordinates) -> bool {
        tile.is_largest_resolution_level()
    }

    fn read_block(&mut self, header: &Header, block: UncompressedBlock) -> UnitResult {
        let line_width = block.index.pixel_size.width();
        let run_width = line_run_pixels::<PxReader::RecursivePixel>(header.channels.list.len())
            .min(line_width);
        if self.line_pixels.len() < run_width {
            self.line_pixels.resize(run_width, PxReader::RecursivePixel::default());
        }

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
        let sink = &self.sink;

        for (y_offset, line_bytes) in byte_lines.enumerate() {
            let y = block.index.pixel_position.y() + y_offset;
            let row = &mut flat[y * storage_width .. (y + 1) * storage_width];

            for x_start in (0..line_width).step_by(run_width) {
                let run_len = run_width.min(line_width - x_start);
                let run = &mut self.line_pixels[..run_len];
                self.pixel_reader
                    .read_pixels(line_bytes, line_width, x_start, run, |px| px);
                for (i, pixel) in run.iter().enumerate() {
                    sink.write(row, x0 + x_start + i, pixel.into_tuple());
                }
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
        let sink = &self.sink;
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
                            // TLS scratch -> reused across chunks on this worker.
                            // Full-width band (x=0)
                            let mut pixels =
                                take_worker_line_scratch::<PxReader::RecursivePixel>(width);

                            for (y_offset, line_bytes) in
                                block.data.chunks_exact(bytes_per_pixel * width).enumerate()
                            {
                                let line = &mut pixels[..width];
                                pixel_reader.read_pixels(line_bytes, width, 0, line, |px| px);
                                let row = &mut this_elements
                                    [y_offset * storage_width .. (y_offset + 1) * storage_width];

                                for (x_offset, pixel) in line.iter().enumerate() {
                                    sink.write(row, x_offset, pixel.into_tuple());
                                }
                            }

                            return_worker_line_scratch(pixels);
                            // same worker recycles decompress buf → next chunk reuses pages
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
impl<PixelStorage, Sink, PxReader, Pixel>
    SpecificChannelsParallelReader<PixelStorage, Sink, PxReader, Pixel>
where
    PxReader: RecursivePixelReader + Sync,
    PxReader::RecursivePixel: IntoTuple<Pixel>,
    PxReader::RecursiveChannelDescriptions: IntoNonRecursive,
    PixelStorage: RowMajorPixelStorage,
    PixelStorage::Element: Send,
    Sink: PixelSink<PixelStorage::Element, Pixel> + Sync,
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
        let sink = &self.sink;
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

                            let mut pixels = take_worker_line_scratch::<PxReader::RecursivePixel>(
                                block_width,
                            );

                            for (y_offset, line_bytes) in block
                                .data
                                .chunks_exact(bytes_per_pixel * block_width)
                                .enumerate()
                            {
                                // Convert outside the lock; only this row's
                                // write contends with tiles that share y.
                                let line = &mut pixels[..block_width];
                                pixel_reader.read_pixels(
                                    line_bytes,
                                    block_width,
                                    0,
                                    line,
                                    |px| px,
                                );

                                let y = y0 + y_offset;
                                let mut row = rows[y].lock().unwrap();
                                for (x_offset, pixel) in line.iter().enumerate() {
                                    sink.write(&mut row, x0 + x_offset, pixel.into_tuple());
                                }
                            }

                            return_worker_line_scratch(pixels);
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

/// Read one channel plane run from `in_bytes` and convert file sample type
fn read_and_convert_all_samples_batched<'t, From, To>(
    mut in_bytes: impl Read,
    out_samples: &mut impl ExactSizeIterator<Item = &'t mut To>,
    convert_batch: fn(&[From], &mut [To]),
) where
    From: Data + Default + Copy + 'static,
    To: 't + Default + Copy + 'static,
{
    let total_sample_count = out_samples.len();
    if total_sample_count == 0 {
        return;
    }

    let run = crate::cpu_cache::l1_resident_count(
        std::mem::size_of::<From>()
            .saturating_add(std::mem::size_of::<To>())
            .max(1),
        2,
    );

    let mut from_buf =
        CONVERT_FROM_SCRATCH.with(|cell| take_tls_any_vec::<From>(cell, run.min(total_sample_count)));
    let mut to_buf =
        CONVERT_TO_SCRATCH.with(|cell| take_tls_any_vec::<To>(cell, run.min(total_sample_count)));

    let len_error_msg = "sample count was miscalculated";
    let byte_error_msg = "error when reading from in-memory slice";

    let mut remaining = total_sample_count;
    while remaining > 0 {
        let n = run.min(remaining);
        if from_buf.len() < n {
            from_buf.resize(n, From::default());
        }
        if to_buf.len() < n {
            to_buf.resize(n, To::default());
        }
        let from = &mut from_buf[..n];
        let to = &mut to_buf[..n];
        Data::read_slice_ne(&mut in_bytes, from).expect(byte_error_msg);
        convert_batch(from, to);
        for sample in to.iter() {
            *out_samples.next().expect(len_error_msg) = *sample;
        }
        remaining -= n;
    }

    CONVERT_FROM_SCRATCH.with(|cell| return_tls_any_vec(cell, from_buf));
    CONVERT_TO_SCRATCH.with(|cell| return_tls_any_vec(cell, to_buf));
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

    #[test]
    fn bulk_f16_to_f32_matches_naive() {
        for total_array_size in [1, 15, 16, 17, 256, 1024, 8192] {
            let input_f16s: Vec<f16> = (0..total_array_size)
                .map(|i| f16::from_f32((i as f32) * 0.01))
                .collect();
            let in_bytes: Vec<u8> =
                input_f16s.iter().flat_map(|v| v.to_ne_bytes()).collect();

            let mut out = vec![0.0f32; total_array_size];
            read_and_convert_all_samples_batched(
                &mut in_bytes.as_slice(),
                &mut out.iter_mut(),
                f32::from_f16s,
            );

            for (i, &bits) in input_f16s.iter().enumerate() {
                assert_eq!(out[i], bits.to_f32(), "index {i}");
            }
        }
    }

    #[test]
    fn bulk_f16_identity_matches() {
        for total_array_size in [1, 64, 4096] {
            let input_f16s: Vec<f16> = (0..total_array_size)
                .map(|i| f16::from_bits(i as u16))
                .collect();
            let in_bytes: Vec<u8> =
                input_f16s.iter().flat_map(|v| v.to_ne_bytes()).collect();

            let mut out = vec![f16::ZERO; total_array_size];
            read_and_convert_all_samples_batched(
                &mut in_bytes.as_slice(),
                &mut out.iter_mut(),
                f16::from_f16s,
            );
            assert_eq!(out, input_f16s);
        }
    }

    #[test]
    fn copy_pixel_sink_writes_identity() {
        let mut row = [(0.0f32, 0.0), (0.0, 0.0), (0.0, 0.0)];
        CopyPixel.write(&mut row, 1, (1.5, 2.5));
        assert_eq!(row[1], (1.5, 2.5));
        // Closure blanket still works
        let convert = |r: &mut [(f32, f32)], x: usize, (a, b): (f32, f32)| {
            r[x] = (a * 2.0, b * 2.0);
        };
        convert.write(&mut row, 0, (1.0, 2.0));
        assert_eq!(row[0], (2.0, 4.0));
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
