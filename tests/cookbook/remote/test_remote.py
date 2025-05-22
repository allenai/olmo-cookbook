from cookbook.remote.base import LocatedPath
from unittest import TestCase
from pathlib import Path
import os


class TestLocatedPath(TestCase):
    def test_located_path_gcs(self):
        self.assertEqual(LocatedPath.from_str("gs://bucket/prefix"), LocatedPath(prot="gs", path="bucket/prefix"))
        self.assertEqual(LocatedPath.from_str("gcs://bucket/prefix"), LocatedPath(prot="gs", path="bucket/prefix"))
        self.assertEqual(
            LocatedPath.from_str("gs://bucket/more/prefix/"), LocatedPath(prot="gs", path="bucket/more/prefix/")
        )

    def test_located_path_s3(self):
        self.assertEqual(LocatedPath.from_str("s3://bucket/prefix"), LocatedPath(prot="s3", path="bucket/prefix"))
        self.assertEqual(
            LocatedPath.from_str("s3n://bucket/prefix/"), LocatedPath(prot="s3", path="bucket/prefix/")
        )

    def test_located_path_weka(self):
        self.assertEqual(
            LocatedPath.from_str("/weka/oe-training-default/prefix"),
            LocatedPath(prot="weka", path="/oe-training-default/prefix"),
        )
        self.assertEqual(
            LocatedPath.from_str("/oe-training-default/prefix/"),
            LocatedPath(prot="weka", path="/oe-training-default/prefix/"),
        )

        with self.assertRaises(ValueError):
            LocatedPath.from_str("weka://non-existent-bucket/prefix")

    def test_located_path_local(self):
        # Test absolute local paths
        self.assertEqual(LocatedPath.from_str("/home/user/data"), LocatedPath(prot="file", path="/home/user/data"))
        self.assertEqual(LocatedPath.from_str("/tmp/data/"), LocatedPath(prot="file", path="/tmp/data/"))

        # Test with Path objects
        self.assertEqual(
            LocatedPath.from_str(Path("/usr/local/bin")), LocatedPath(prot="file", path="/usr/local/bin")
        )

        # Test single-level paths
        self.assertEqual(LocatedPath.from_str("/home"), LocatedPath(prot="file", path="/home"))

    def test_located_path_to_str(self):
        # Test conversion back to string
        path = LocatedPath(prot="file", path="home/user/data")
        self.assertEqual(path.local, Path("home/user/data"))

        path_with_trailing_slash = LocatedPath(prot="file", path="tmp/data/")
        self.assertEqual(path_with_trailing_slash.local, Path("tmp/data/"))

    def test_located_path_invalid(self):
        with self.assertRaises(ValueError):
            LocatedPath.from_str("azure://bucket/prefix")

    def test_local_command(self):
        with self.assertRaises(ValueError):
            LocatedPath.from_str("s3://bucket/prefix").local

        with self.assertRaises(ValueError):
            LocatedPath.from_str("gs://bucket/prefix").local

        self.assertEqual(LocatedPath.from_str("file://home/user/data").local, Path("/home/user/data"))
        self.assertEqual(LocatedPath.from_str("/home/user/data").local, Path("/home/user/data"))
        self.assertEqual(
            LocatedPath.from_str("weka://oe-data-default/prefix").local, Path("/oe-data-default/prefix")
        )
        self.assertEqual(LocatedPath.from_str("/oe-data-default/prefix").local, Path("/oe-data-default/prefix"))
        self.assertEqual(
            LocatedPath.from_str("/weka/oe-training-default/prefix").local, Path("/oe-training-default/prefix")
        )

    def test_remote_command(self):
        with self.assertRaises(ValueError):
            LocatedPath.from_str("file://home/user/data").remote

        with self.assertRaises(ValueError):
            LocatedPath.from_str("/home/user/data").remote

        self.assertEqual(LocatedPath.from_str("s3://bucket/prefix").remote, "s3://bucket/prefix")
        self.assertEqual(LocatedPath.from_str("gs://bucket/prefix").remote, "gs://bucket/prefix")
        self.assertEqual(
            LocatedPath.from_str("/oe-training-default/prefix").remote, "weka://oe-training-default/prefix"
        )
        self.assertEqual(
            LocatedPath.from_str("/weka/oe-training-default/prefix").remote, "weka://oe-training-default/prefix"
        )
