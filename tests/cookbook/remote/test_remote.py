from cookbook.remote.base import LocatedPath
from unittest import TestCase
from pathlib import Path
import os


class TestLocatedPath(TestCase):
    def test_located_path_gcs(self):
        self.assertEqual(LocatedPath.from_str("gs://bucket/prefix"), LocatedPath(prot="gs", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("gcs://bucket/prefix"), LocatedPath(prot="gs", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("gs://bucket/more/prefix/"), LocatedPath(prot="gs", bucket="bucket", prefix="more/prefix/"))


    def test_located_path_s3(self):
        self.assertEqual(LocatedPath.from_str("s3://bucket/prefix"), LocatedPath(prot="s3", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("s3n://bucket/prefix/"), LocatedPath(prot="s3", bucket="bucket", prefix="prefix/"))


    def test_located_path_weka(self):
        self.assertEqual(LocatedPath.from_str("/weka/oe-training-default/prefix"), LocatedPath(prot="weka", bucket="oe-training-default", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("/oe-training-default/prefix/"), LocatedPath(prot="weka", bucket="oe-training-default", prefix="prefix/"))

        with self.assertRaises(ValueError):
            LocatedPath.from_str("weka://non-existent-bucket/prefix")

    def test_located_path_local(self):
        # Test absolute local paths
        self.assertEqual(LocatedPath.from_str("/home/user/data"), LocatedPath(prot="file", bucket="home", prefix="user/data"))
        self.assertEqual(LocatedPath.from_str("/tmp/data/"), LocatedPath(prot="file", bucket="tmp", prefix="data/"))

        # Test with Path objects
        self.assertEqual(LocatedPath.from_str(Path("/usr/local/bin")), LocatedPath(prot="file", bucket="usr", prefix="local/bin"))

        # Test single-level paths
        self.assertEqual(LocatedPath.from_str("/home"), LocatedPath(prot="file", bucket="home", prefix=""))

    def test_located_path_to_str(self):
        # Test conversion back to string
        path = LocatedPath(prot="file", bucket="home", prefix="user/data")
        self.assertEqual(path.to_str(), "file://home/user/data")

        path_with_trailing_slash = LocatedPath(prot="file", bucket="tmp", prefix="data/")
        self.assertEqual(path_with_trailing_slash.to_str(), "file://tmp/data/")

    def test_located_path_full_property(self):
        # Test the full property
        path = LocatedPath(prot="file", bucket="home", prefix="user/data")
        self.assertEqual(path.full, Path("/home/user/data"))

        path_with_trailing_slash = LocatedPath(prot="gs", bucket="bucket", prefix="prefix/")
        self.assertEqual(path_with_trailing_slash.full, Path("/bucket/prefix/"))

    def test_located_path_invalid(self):
        with self.assertRaises(ValueError):
            LocatedPath.from_str("azure://bucket/prefix")
        with self.assertRaises(ValueError):
            LocatedPath.from_str("/weka/non-existent-bucket/prefix")
