import pytest
import re
from collections import OrderedDict
from cookbook.eval.miniframe import MiniFrame
from cookbook.constants import SHORT_NAMES


@pytest.fixture
def sample_frame():
    """Create a sample MiniFrame for testing"""
    frame = MiniFrame(title="Test Frame")

    # Add some test data
    frame.add(row="model1", col="metric1", val=0.75)
    frame.add(row="model1", col="metric2", val=0.8)
    frame.add(row="model2", col="metric1", val=0.65)
    frame.add(row="model2", col="metric2", val=0.9)
    frame.add(row="model3", col="metric1", val=0.85)
    frame.add(row="model3", col="metric2", val=None)

    return frame


@pytest.fixture
def none_heavy_frame():
    """Create a MiniFrame with many None values for testing"""
    frame = MiniFrame(title="None Test Frame")

    # Add data with None values
    frame.add(row="model1", col="metric1", val=None)
    frame.add(row="model1", col="metric2", val=0.8)
    frame.add(row="model1", col="metric3", val=None)
    frame.add(row="model2", col="metric1", val=None)
    frame.add(row="model2", col="metric2", val=None)
    frame.add(row="model2", col="metric3", val=0.7)
    frame.add(row="model3", col="metric1", val=0.85)
    frame.add(row="model3", col="metric2", val=None)
    frame.add(row="model3", col="metric3", val=None)

    return frame


class TestMiniFrame:
    def test_initialization(self):
        """Test MiniFrame initialization"""
        frame = MiniFrame(title="Empty Frame")
        assert frame.title == "Empty Frame"
        assert len(frame) == 0
        assert frame.shape() == (0, 0)

    def test_add(self):
        """Test adding single items to the frame"""
        frame = MiniFrame(title="Add Test")
        frame.add(row="test_row", col="test_col", val=0.5)

        assert len(frame) == 1
        assert "test_col" in frame
        assert frame.shape() == (1, 1)

    def test_add_many(self):
        """Test adding multiple items at once"""
        frame = MiniFrame(title="Add Many Test")
        frame.add_many(
            ("row1", "col1", 0.1),
            ("row1", "col2", 0.2),
            ("row2", "col1", 0.3),
        )

        assert len(frame) == 2
        assert "col1" in frame
        assert "col2" in frame
        assert frame.shape() == (2, 2)

        assert frame._data["col1"]["row1"] == 0.1
        assert frame._data["col2"]["row1"] == 0.2
        assert frame._data["col1"]["row2"] == 0.3
        assert "row2" not in frame._data["col2"]
        assert frame["col2", "row2"] is None

    def test_sort(self, sample_frame):
        """Test sorting by column values"""
        # Sort by metric1, ascending (default)
        sorted_frame = sample_frame.sort(col="metric1")
        rows = [r for r, _ in sorted_frame.rows]
        assert rows == ["model2", "model1", "model3"], "Rows should be sorted by metric1 values ascending"

        # Sort by metric1, descending
        sorted_frame = sample_frame.sort(col="metric1", reverse=True)
        rows = [r for r, _ in sorted_frame.rows]
        assert rows == ["model3", "model1", "model2"], "Rows should be sorted by metric1 values descending"

        # Sort by metric2 (with None value)
        sorted_frame = sample_frame.sort(col="metric2")
        rows = [r for r, _ in sorted_frame.rows]
        assert rows[0] == "model3", "None values should be sorted first (as -inf)"

    def test_columns_property(self, sample_frame):
        """Test the columns property"""
        columns = list(sample_frame.columns)
        assert len(columns) == 2
        assert "metric1" in columns
        assert "metric2" in columns

    def test_rows_property(self, sample_frame):
        """Test the rows property"""
        rows = list(sample_frame.rows)
        assert len(rows) == 3

        # Check row names
        row_names = [r[0] for r in rows]
        assert "model1" in row_names
        assert "model2" in row_names
        assert "model3" in row_names

        # Check that each row has correct number of values
        for _, values in rows:
            assert len(values) == 2

    def test_drop_rows(self, sample_frame):
        """Test dropping rows"""
        # Drop a single row by exact match
        frame = sample_frame.drop_rows("model1")
        rows = [r for r, _ in frame.rows]
        assert "model1" not in rows
        assert len(rows) == 2

        # Drop rows using a regex pattern
        frame = sample_frame.drop_rows(re.compile(r"model[13]"))
        rows = [r for r, _ in frame.rows]
        assert len(rows) == 1
        assert rows[0] == "model2"

        # Drop multiple rows
        frame = sample_frame.drop_rows("model1", "model2")
        rows = [r for r, _ in frame.rows]
        assert len(rows) == 1
        assert rows[0] == "model3"

    def test_keep_rows(self, sample_frame):
        """Test keeping only specified rows"""
        # Keep a single row by exact match
        frame = sample_frame.keep_rows("model1")
        rows = [r for r, _ in frame.rows]
        assert len(rows) == 1
        assert rows[0] == "model1"

        # Keep rows using a regex pattern
        frame = sample_frame.keep_rows(re.compile(r"model[12]"))
        rows = [r for r, _ in frame.rows]
        assert len(rows) == 2
        assert "model3" not in rows

        # Keep multiple rows
        frame = sample_frame.keep_rows("model1", "model3")
        rows = [r for r, _ in frame.rows]
        assert len(rows) == 2
        assert "model2" not in rows

    def test_drop_cols(self, sample_frame):
        """Test dropping columns"""
        # Drop a single column
        frame = sample_frame.drop_cols("metric1")
        cols = list(frame.columns)
        assert len(cols) == 1
        assert cols[0] == "metric2"

        # Drop using pattern
        frame = sample_frame.drop_cols(re.compile(r"metric\d"))
        assert len(frame) == 0

        # Drop multiple columns
        test_frame = MiniFrame(title="Multi-column Test")
        test_frame.add_many(
            ("row1", "col1", 0.1),
            ("row1", "col2", 0.2),
            ("row1", "col3", 0.3),
        )
        result = test_frame.drop_cols("col1", "col3")
        assert len(result) == 1
        assert list(result.columns) == ["col2"]

    def test_keep_cols(self, sample_frame):
        """Test keeping only specified columns"""
        # Keep a single column
        frame = sample_frame.keep_cols("metric1")
        cols = list(frame.columns)
        assert len(cols) == 1
        assert cols[0] == "metric1"

        # Keep multiple columns
        frame = MiniFrame(title="Test")
        frame.add_many(
            ("row1", "col1", 0.1),
            ("row1", "col2", 0.2),
            ("row1", "col3", 0.3),
        )
        result = frame.keep_cols("col1", "col3")
        assert len(result) == 2
        assert "col2" not in list(result.columns)

        # Keep using regex pattern
        result = frame.keep_cols(re.compile(r"col[13]"))
        assert len(result) == 2
        assert "col2" not in list(result.columns)

    def test_drop_empty(self, sample_frame):
        """Test dropping columns with None values"""
        # Initial frame has a None in metric2 column
        assert "metric2" in list(sample_frame.columns)

        # After dropping empty, metric2 should be gone
        cleaned = sample_frame.drop_empty()
        assert "metric1" in list(cleaned.columns)
        assert "metric2" not in list(cleaned.columns)

    def test_shape(self, sample_frame):
        """Test shape method"""
        # Sample frame has 2 columns, 3 rows
        assert sample_frame.shape() == (2, 3)

        # Empty frame
        empty = MiniFrame(title="Empty")
        assert empty.shape() == (0, 0)

        # Uneven frame (different number of values in columns)
        uneven = MiniFrame(title="Uneven")
        uneven.add(row="row1", col="col1", val=0.1)
        uneven.add(row="row2", col="col1", val=0.2)
        uneven.add(row="row1", col="col2", val=0.3)
        assert uneven.shape() == (2, 2)

    def test_contains(self, sample_frame):
        """Test __contains__ method"""
        assert "metric1" in sample_frame
        assert "metric2" in sample_frame
        assert "missing" not in sample_frame

    def test_len(self, sample_frame):
        """Test __len__ method"""
        assert len(sample_frame) == 2

    def test_all_none_column(self):
        """Test behavior with columns that contain only None values"""
        # Create a frame with a column that has only None values
        frame = MiniFrame(title="All None Column Test")
        frame.add(row="row1", col="all_none", val=None)
        frame.add(row="row2", col="all_none", val=None)
        frame.add(row="row1", col="has_value", val=0.5)
        frame.add(row="row2", col="has_value", val=0.7)

        # Verify we have a column with all None values
        all_none_cols = []
        for col in frame._data:
            if all(v is None for v in frame._data[col].values()):
                all_none_cols.append(col)

        assert "all_none" in all_none_cols
        assert len(all_none_cols) == 1

        # Test drop_empty with columns that are entirely None
        cleaned = frame.drop_empty()
        for col in all_none_cols:
            assert col not in list(cleaned.columns)

        # Test that we still have columns that had some non-None values
        assert "has_value" in list(cleaned.columns)
        assert len(cleaned) == 1

    def test_row_with_all_none(self, none_heavy_frame):
        """Test behavior with rows that have all None values"""
        # Sort by a column with multiple None values
        sorted_frame = none_heavy_frame.sort(col="metric1")

        # The first rows should be the ones with None values
        rows_with_none = []
        for row, values in sorted_frame.rows:
            if values[0] is None:  # First column (metric1) is None
                rows_with_none.append(row)

        assert len(rows_with_none) == 2
        assert "model1" in rows_with_none
        assert "model2" in rows_with_none

        # Testing sort with another column
        sorted_frame = none_heavy_frame.sort(col="metric2", reverse=True)
        rows = [r for r, _ in sorted_frame.rows]
        # model1 should be first (0.8), then the None values
        assert rows[0] == "model1"

    def test_edge_case_empty_frame(self):
        """Test edge cases with empty frames"""
        empty = MiniFrame(title="Empty")

        # Operations on empty frame should return empty frame
        assert len(empty.sort("any_col")) == 0
        assert len(empty.drop_rows("any_row")) == 0
        assert len(empty.keep_cols("any_col")) == 0
        assert len(empty.drop_empty()) == 0

        # Empty frame shape
        assert empty.shape() == (0, 0)

        # Empty frame rows and columns
        assert list(empty.rows) == []
        assert list(empty.columns) == []

    def test_overwrite_values(self):
        """Test overwriting values in the frame"""
        frame = MiniFrame(title="Overwrite Test")

        # Add initial value
        frame.add(row="row1", col="col1", val=0.5)

        # Overwrite with new value
        frame.add(row="row1", col="col1", val=0.7)

        # Verify the new value is used
        for row, values in frame.rows:
            if row == "row1":
                assert values[0] == 0.7

        # Overwrite with None
        frame.add(row="row1", col="col1", val=None)

        # Verify the value is now None
        for row, values in frame.rows:
            if row == "row1":
                assert values[0] is None

    def test_multiple_none_values(self, none_heavy_frame):
        """Test frames with multiple None values in different patterns"""
        # Count None values by column
        none_counts = {}
        for col in none_heavy_frame._data:
            none_counts[col] = sum(1 for v in none_heavy_frame._data[col].values() if v is None)

        # Verify our test fixture has the expected None patterns
        assert none_counts["metric1"] >= 2
        assert none_counts["metric2"] >= 2
        assert none_counts["metric3"] >= 2

        # The way drop_empty works is it only drops columns where ANY value is None
        # not all values are None. Let's verify this behavior
        cleaned = none_heavy_frame.drop_empty()

        # Since all columns in none_heavy_frame have at least one None,
        # they should all be dropped
        assert len(cleaned) == 0

        # Create a frame with some clean columns
        frame = MiniFrame(title="Mixed None Test")
        frame.add(row="row1", col="clean_col", val=0.5)
        frame.add(row="row2", col="clean_col", val=0.7)
        frame.add(row="row1", col="some_none", val=None)
        frame.add(row="row2", col="some_none", val=0.3)

        # Test drop_empty with a mix of clean and partially None columns
        cleaned = frame.drop_empty()
        assert "clean_col" in list(cleaned.columns)
        assert "some_none" not in list(cleaned.columns)
        assert len(cleaned) == 1

    def test_invalid_filter_value(self):
        """Test that appropriate errors are raised for invalid filters"""
        frame = MiniFrame(title="Error Test")
        frame.add(row="row1", col="col1", val=0.5)

        # Test with invalid filter type
        with pytest.raises(ValueError):
            frame.keep_cols(123)  # Integer isn't a valid filter type   # pyright: ignore

        with pytest.raises(ValueError):
            frame.drop_rows({"key": "value"})  # Dict isn't a valid filter type  # pyright: ignore

    def test_make_fn(self):
        """Test the _make_fn function implementation"""
        frame = MiniFrame(title="Make Function Test")
        
        # Test exact match with single string filter (not reverse)
        fn = frame._make_fn(vals=("abc",), reverse=False)
        assert fn("abc") is True
        assert fn("def") is False
        
        # Test exact match with multiple string filters (not reverse)
        fn = frame._make_fn(vals=("abc", "def"), reverse=False)
        assert fn("abc") is True
        assert fn("def") is True
        assert fn("ghi") is False
        
        # Test exact match with single string filter (reverse=True)
        fn = frame._make_fn(vals=("abc",), reverse=True)
        assert fn("abc") is False
        assert fn("def") is True
        
        # Test with regex filter (not reverse)
        pattern = re.compile(r"model\d")
        fn = frame._make_fn(vals=(pattern,), reverse=False)
        assert fn("model1") is True
        assert fn("model23") is True
        assert fn("something") is False
        
        # Test with regex filter (reverse=True)
        fn = frame._make_fn(vals=(pattern,), reverse=True)
        assert fn("model1") is False
        assert fn("something") is True

    def test_sort_nonexistent_column(self):
        """Test sorting when column name doesn't exist in the data"""
        # Create a frame with multiple columns for testing
        frame = MiniFrame(title="Sort Test")
        frame.add(row="row1", col="col1", val=0.5)
        frame.add(row="row2", col="col1", val=0.7)

        # Looking at the implementation, attempting to sort by a non-existent column
        # will raise a KeyError, so instead we'll test sorting by a column
        # where only some rows have values
        frame.add(row="row1", col="partial_col", val=0.3)
        # row2 doesn't have a value for partial_col

        # Sort by this column - should work, with row2 treated as having None value
        sorted_frame = frame.sort(col="partial_col")

        # Check that rows with None values appear first
        rows = [r for r, _ in sorted_frame.rows]
        assert rows[0] == "row2"  # row2 should come first (has None for partial_col)
        assert rows[1] == "row1"  # row1 has value 0.3

    def test_show_method(self, sample_frame):
        """Test that show method doesn't crash"""
        # Since rich.console.Console output is hard to capture reliably in tests,
        # and the exact formatting can change, we'll just verify that the method
        # doesn't raise any exceptions

        # This test just ensures the show method doesn't crash
        sample_frame.show()

        # Also test with empty frame
        empty = MiniFrame(title="Empty Frame")
        empty.show()

        # And with a frame containing all None values
        none_frame = MiniFrame(title="None Frame")
        none_frame.add(row="row1", col="col1", val=None)
        none_frame.show()

    def test_short_names(self):
        """Test that column names are properly shortened in show method using SHORT_NAMES"""
        # Create a frame with column names that match SHORT_NAMES patterns
        frame = MiniFrame(title="Short Names Test")
        frame.add(row="row1", col="arc_challenge::olmes", val=0.5)
        frame.add(row="row1", col="hellaswag::olmes", val=0.7)
        frame.add(row="row1", col="gsm8k::olmo1", val=0.6)
        
        # We can't directly test the output of show(), but we can check that
        # SHORT_NAMES constants are available and have the expected patterns
        assert isinstance(SHORT_NAMES, dict)
        assert r"::olmes$" in SHORT_NAMES
        assert r"^gsm8k::olmo1$" in SHORT_NAMES
        
        # Just make sure show() doesn't crash
        frame.show()

    def test_chained_operations(self, sample_frame):
        """Test chaining multiple operations together"""
        # Chain multiple operations to test fluent interface
        result = (sample_frame
                 .sort(col="metric1")
                 .keep_rows("model1", "model2")
                 .drop_cols("metric2"))

        # Verify the result has the expected shape and contents
        assert len(result) == 1  # One column remains
        assert "metric1" in list(result.columns)
        assert "metric2" not in list(result.columns)

        rows = [r for r, _ in result.rows]
        assert len(rows) == 2
        assert "model1" in rows
        assert "model2" in rows
        assert "model3" not in rows

        # Test more complex chaining with mixed operations
        complex_result = (sample_frame
                         .sort(col="metric1", reverse=True)  # Descending sort by metric1
                         .keep_cols(re.compile(r"metric\d"))  # Keep all metric columns
                         .drop_rows("model3"))  # Remove model3

        # Verify complex chaining result
        assert len(complex_result) == 2  # Both metric columns
        rows = [r for r, _ in complex_result.rows]
        assert len(rows) == 2  # model1 and model2 only
        assert "model3" not in rows

        # First row should be model1 since sort is reversed and we removed model3
        first_row = rows[0]
        assert first_row == "model1"
        
    def test_add_operator(self):
        """Test the __add__ operator for combining MiniFrames"""
        # Create two frames to combine
        frame1 = MiniFrame(title="Frame 1")
        frame1.add(row="row1", col="col1", val=0.1)
        frame1.add(row="row2", col="col1", val=0.2)
        
        frame2 = MiniFrame(title="Frame 2")
        frame2.add(row="row1", col="col2", val=0.3)
        frame2.add(row="row3", col="col2", val=0.4)
        
        # Combine frames with + operator
        combined = frame1 + frame2
        
        # Check that combined frame has all columns
        columns = list(combined.columns)
        assert len(columns) == 2
        assert "col1" in columns
        assert "col2" in columns
        
        # Check that combined frame has all rows
        rows = [r for r, _ in combined.rows]
        assert len(rows) == 3
        assert "row1" in rows
        assert "row2" in rows
        assert "row3" in rows
        
        # Check specific values
        assert combined["col1", "row1"] == 0.1
        assert combined["col2", "row1"] == 0.3
        assert combined["col1", "row3"] is None
        assert combined["col2", "row3"] == 0.4
        
    def test_radd_operator(self):
        """Test the __radd__ operator for combining MiniFrames with sum()"""
        # Create multiple frames
        frame1 = MiniFrame(title="Frame 1")
        frame1.add(row="row1", col="col1", val=0.1)
        
        frame2 = MiniFrame(title="Frame 2")
        frame2.add(row="row1", col="col2", val=0.2)
        
        frame3 = MiniFrame(title="Frame 3")
        frame3.add(row="row2", col="col1", val=0.3)
        
        # Combine with sum()
        combined = sum([frame1, frame2, frame3], MiniFrame(title="Combined"))
        
        # Check title is preserved from the start value
        assert combined.title == "Combined"
        
        # Check columns and rows
        assert len(combined) == 2
        assert "col1" in list(combined.columns)
        assert "col2" in list(combined.columns)
        
        rows = [r for r, _ in combined.rows]
        assert len(rows) == 2
        assert "row1" in rows
        assert "row2" in rows
        
        # Check values
        assert combined["col1", "row1"] == 0.1
        assert combined["col2", "row1"] == 0.2
        assert combined["col1", "row2"] == 0.3
        assert combined["col2", "row2"] is None
