class DatasetError(Exception):
    """Base exception for safe dataset operations."""


class DatasetAcquisitionError(DatasetError):
    """Raised when a dataset cannot be acquired safely."""


class DownloadTimeoutError(DatasetAcquisitionError):
    """Raised when a dataset request exceeds its timeout."""


class DownloadSizeError(DatasetAcquisitionError):
    """Raised when a download exceeds the configured byte limit."""


class ChecksumMismatchError(DatasetAcquisitionError):
    """Raised when an explicitly expected checksum does not match."""


class InvalidArchiveError(DatasetAcquisitionError):
    """Raised when an archive is corrupt or unsafe."""


class ExistingDataError(DatasetAcquisitionError):
    """Raised when replacement would require an explicit force flag."""


class DatasetRecordError(DatasetError):
    """Raised with source context when a dataset record cannot be normalized."""


class AmbiguousWorldAssumptionError(DatasetRecordError):
    """Raised when a raw label cannot be mapped safely under its semantics."""


class DuplicateExampleError(DatasetRecordError):
    """Raised when a deterministic normalized example ID is repeated."""


class SamplingError(DatasetError):
    """Raised when a requested sample is unsafe or impossible."""
