import abc
import os
from typing import List, Optional, Tuple

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError


class InvalidGCPBucketFilePath(Exception):
    pass


def is_gcp_path(filepath: str) -> bool:
    return filepath.startswith("gs://")


class Filepath(abc.ABC):
    @abc.abstractmethod
    def delete(self) -> Tuple[bool, str]:
        pass

    @abc.abstractmethod
    def exists(self) -> Tuple[bool, str]:
        pass


def from_filepath(filepath: str) -> Filepath:
    if is_gcp_path(filepath):
        return GCPBucketFilepath.from_gcp_filepath(filepath)
    return LocalFilepath(filepath)


class LocalFilepath(Filepath):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def delete(self) -> Tuple[bool, str]:
        try:
            os.remove(self.filepath)
        except FileNotFoundError:
            return (False, f"File not found {self.filepath}")
        except PermissionError:
            return (False, f"Permission error for {self.filepath}")
        return (True, "")

    def exists(self) -> Tuple[bool, str]:
        return (os.path.exists(self.filepath), "")


class GCPBucketFilepath(Filepath):
    def __init__(self, gcp_bucket_name: str, filepath: str):
        self.gcp_bucket_name = gcp_bucket_name
        self.filepath = filepath

        if is_gcp_path(self.gcp_bucket_name):
            self.gcp_bucket_name = self.gcp_bucket_name[5:]

    def get_gcp_filepath(self) -> str:
        return f"gs://{self.gcp_bucket_name}/{self.filepath}"

    @classmethod
    def from_gcp_filepath(cls, gcp_filepath: str):
        if not gcp_filepath.startswith("gs://"):
            raise InvalidGCPBucketFilePath(
                f"GCP filepath {gcp_filepath} does not start with gs://"
            )

        if len(gcp_filepath.split("/")) <= 1:
            raise InvalidGCPBucketFilePath(
                f"GCP filepath {gcp_filepath} does not contain at least two /"
            )

        return cls(
            gcp_bucket_name=gcp_filepath[5:].split("/")[0],
            filepath="/".join(gcp_filepath[5:].split("/")[1:]),
        )

    def upload(
        self, filepath_to_local_file: str, bytes_format: bool = False
    ) -> Tuple[bool, str]:
        error_msg = (
            "%s when attempting to upload the local file "
            f"{filepath_to_local_file} to {self.get_gcp_filepath()} "
            "exact error: %s"
        )

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            blob = bucket.blob(self.filepath)
            with open(
                filepath_to_local_file, "rb" if bytes_format else "r"
            ) as local_file:
                blob.upload_from_file(local_file)
        except GoogleCloudError as google_cloud_exception:
            return (
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
            )
        except Exception as exception:
            return (False, error_msg % ("Uncaught exception", str(exception)))
        return (True, "")

    def upload_from_string(self, content: str) -> Tuple[bool, str]:
        error_msg = (
            f"%s when attempting to upload to {self.get_gcp_filepath()} "
            "exact error: %s"
        )

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            blob = bucket.blob(self.filepath)
            blob.upload_from_string(content)
        except GoogleCloudError as google_cloud_exception:
            return (
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
            )
        except Exception as exception:
            return (False, error_msg % ("Uncaught exception", str(exception)))
        return (True, "")

    def download_to_file(self, filepath_to_local_file: str) -> Tuple[bool, str]:
        return self.download(filepath_to_local_file)[1:]

    def download_to_string(self) -> Tuple[str, bool, str]:
        return self.download()

    def download(
        self, filepath_to_local_file: Optional[str] = None
    ) -> Tuple[str, bool, str]:
        error_msg = (
            "%s when attempting to download the "
            f"gcp file {self.get_gcp_filepath()}"
            + (
                f" to {filepath_to_local_file}"
                if filepath_to_local_file is not None
                else ""
            )
            + ", exact error: %s"
        )

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            blob = bucket.blob(self.filepath)
            if filepath_to_local_file is not None:
                blob.download_to_filename(filepath_to_local_file)
            else:
                file_content = blob.download_as_string()
                return (file_content, True, "")
        except GoogleCloudError as google_cloud_exception:
            return (
                "",
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
            )
        except Exception as exception:
            return ("", False, error_msg % ("Uncaught Exception", str(exception)))
        return ("", True, "")

    def delete(self) -> Tuple[bool, str]:
        error_msg = (
            "%s when attempting to delete the "
            f"gcp file {self.get_gcp_filepath()} , exact error: %s"
        )
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcp_bucket_name)
            bucket.delete_blob(self.filepath)
            return (True, "")
        except GoogleCloudError as google_cloud_exception:
            return (
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
            )
        except Exception as exception:
            return (False, error_msg % ("Uncaught Exception", str(exception)))

    def listdir(
        self, max_dir_level: Optional[int] = 0
    ) -> Tuple[bool, str, Optional[List[str]]]:
        """This is the same as os.listdir when max_dir_level is 0.

        Parameters:
            max_dir_level (int): maximum level of subdirectories to browse.
                If None, all subdirectories are browsed.
        """
        error_msg = (
            "%s when attempting to pinging the gcp bucket "
            f"{self.gcp_bucket_name} to list "
            f"elements in the folder {self.filepath}. "
            "Exact error: %s."
        )

        try:
            storage_client = storage.Client()
            filepath = (
                self.filepath if self.filepath.endswith("/") else f"{self.filepath}/"
            )
            nb_char_filepath = len(filepath)
            result = set()

            for blob in storage_client.list_blobs(
                self.gcp_bucket_name,
                prefix=filepath,
            ):
                gcp_filepath_wo_prefix = blob.name[nb_char_filepath:]

                if not bool(gcp_filepath_wo_prefix):
                    continue

                dir_level = gcp_filepath_wo_prefix.count("/")
                is_folder = gcp_filepath_wo_prefix.endswith("/")

                if is_folder:
                    dir_level -= 1

                max_subdir_level = (
                    min(dir_level, max_dir_level)
                    if max_dir_level is not None
                    else dir_level
                )

                if not (max_subdir_level == dir_level and not is_folder):
                    max_subdir_level += 1
                else:
                    result.add(gcp_filepath_wo_prefix)

                for sub_dir_level in range(max_subdir_level):
                    higher_level_folder = "/".join(
                        gcp_filepath_wo_prefix.split("/")[: (sub_dir_level + 1)]
                    )
                    result.add(f"{higher_level_folder}/")

            return (True, "", list(result))
        except GoogleCloudError as google_cloud_exception:
            return (
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
                None,
            )
        except Exception as exception:
            return (False, error_msg % ("Uncaught Exception", str(exception)), None)

    def exists(self) -> Tuple[bool, str]:
        error_msg = (
            "%s when attempting to pinging the gcp bucket "
            f"{self.gcp_bucket_name} to check if the file "
            f"{self.filepath} exists, "
            "exact error: %s"
        )

        try:
            storage_client = storage.Client()
            match_found = (
                len(
                    list(
                        storage_client.list_blobs(
                            self.gcp_bucket_name,
                            prefix=self.filepath,
                        )
                    )
                )
                == 1
            )
        except GoogleCloudError as google_cloud_exception:
            return (
                False,
                error_msg % ("GoogleCloudError caught", str(google_cloud_exception)),
            )
        except Exception as exception:
            return (False, error_msg % ("Uncaught Exception", str(exception)))
        return (match_found, "")
