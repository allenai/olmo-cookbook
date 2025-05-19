from cookbook.remote.base import LocatedPath
from unittest import TestCase


class TestLocatedPath(TestCase):
    def test_located_path_gcs(self):
        self.assertEqual(LocatedPath.from_str("gs://bucket/prefix"), LocatedPath(prot="gs", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("gcs://bucket/prefix"), LocatedPath(prot="gs", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("gs://bucket/more/prefix/"), LocatedPath(prot="gs", bucket="bucket", prefix="more/prefix/"))


    def test_located_path_s3(self):
        self.assertEqual(LocatedPath.from_str("s3://bucket/prefix"), LocatedPath(prot="s3", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("s3n://bucket/prefix/"), LocatedPath(prot="s3", bucket="bucket", prefix="prefix/"))


    def test_located_path_weka(self):
        self.assertEqual(LocatedPath.from_str("weka://bucket/prefix"), LocatedPath(prot="weka", bucket="bucket", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("/weka/oe-training-default/prefix"), LocatedPath(prot="weka", bucket="oe-training-default", prefix="prefix"))
        self.assertEqual(LocatedPath.from_str("/oe-training-default/prefix/"), LocatedPath(prot="weka", bucket="oe-training-default", prefix="prefix/"))

    def test_located_path_invalid(self):
        with self.assertRaises(ValueError):
            LocatedPath.from_str("bucket/prefix")
        with self.assertRaises(ValueError):
            LocatedPath.from_str("azure://bucket/prefix")
        with self.assertRaises(ValueError):
            LocatedPath.from_str("/weka/non-existent-bucket/prefix")
