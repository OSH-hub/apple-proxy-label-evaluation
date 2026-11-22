# Source I/O scheduling amendment

The first v4 extraction attempt read approximately two source images per second from SMB-mounted ZIP files. Its eight workers waited for network filesystem pages; model fitting had not begun. All nine source ZIP files total approximately 544 GB, whereas selected entries represent a much smaller subset. Copying the complete archives was therefore avoided.

Only our extraction process was stopped. Its source code and progress log are archived as `rebuild_source_crops_initial_io.py` and `source_crops_initial_io.log`. Existing v4 attempt crops are preserved and reused only if their bytes exactly equal fresh regeneration. A difference is a hard error. No v2/v3 or original source file was changed.

The replacement schedules one sequential queue per archive, sorting reads by ZIP header offset. A single bounded positional read retrieves the compressed payload and maximum ZIP header allowance. Standard raw DEFLATE or stored-byte decoding is checked against each central-directory CRC32 and uncompressed length. Image decoding, JSON checks, labels, bbox geometry, padding, JPEG encoding, sample membership and original metadata order are unchanged. Nine queues run concurrently. The initial code had already checked ZIP CRCs through the standard library; the new path checks them explicitly. Model results were not inspected when making this I/O-only change.

A read-only 20-image timing check on different adjacent source entries took 2.32 seconds using standard ZipFile reading and 1.62 seconds using positional reads; this small scheduling diagnostic is not a scientific result or a full speed guarantee. The complete source audit remains required before training.
