// AC unRLE experiments only (not in decode). `v4_experiment`: AVX-512 loses
// (cache-vs-branch findings). `alt_experiment`: fused/pipe still open; branchless
// shipped (`dwa-un-rle-ac-branchless-findings`).
pub(crate) mod alt_experiment;
pub(crate) mod v4_experiment;
