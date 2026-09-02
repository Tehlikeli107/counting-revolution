# V119 usage

Run:

`177_COUNTING_REVOLUTION_V119_INDEPENDENT_RAW_FRONTIER_REPLICATION.bat`

## What it does

- verifies the V63 hard-set certificate and V102 obstruction certificate;
- locates or downloads the exact official `graph10.g6`;
- independently computes all ten atomic deleted-card fields for every one of
  the 12,005,168 catalog graphs;
- stores them in a resumable local int32 memmap;
- computes eight theorem-critical full-catalog partitions;
- independently reconstructs the zero-bit minimum without V73;
- directly rechecks all V102 pair/single witness signatures and non-isomorphism;
- rechecks the explicit mod-127 one-bit channel;
- emits the exact frontier `[5,3,3,2,2,2,1]`.

## Important

The `.v119_work` directory is intentionally large and persistent. Do not delete
it before the result is audited; it is the reusable raw-replication workspace.

The result ZIP itself is compact and is named:

`V119_INDEPENDENT_RAW_FRONTIER_REPLICATION_RESULT_YYYYMMDD_HHMMSS.zip`
