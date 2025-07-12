"""This module was written mainly by Claude Sonnet 4 and then modified.

Constructing pdfs is ugly. This module is a quite dirty hack X-(
Only really tested for font size 6
"""

from collections.abc import Sequence
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors, pagesizes
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT


class DataFrameToPDF:
    def __init__(
        self,
        filename: str | Path,
        pagesize="A4",
        orientation="portrait",
        min_font_size=6,
        max_font_size=12,
        pagination_strategy="rows_first",
        min_width_factor: float | Sequence[float] = 1,
    ):
        """
        Initialize the PDF generator with auto-scaling parameters.

        Args:
            filename (str | Path): Output PDF filename
            pagesize: Page size (A4, letter, etc.)
            orientation (str): 'portrait' or 'landscape'
            min_font_size (int): Minimum font size for scaling
            max_font_size (int): Maximum font size for scaling
            pagination_strategy (str): 'rows_first' or 'columns_first'
                - 'rows_first': Print all rows for first set of columns, then next set
                - 'columns_first': Print all columns for first set of rows, then next set
            min_width_factor (float | Sequence[float]): Minimum width factor for column compression
                - 1.0: No compression (full content width)
                - 0.5: Allow compression to 50% of required width
                - 0.25: Allow compression to 25% of required width
        """
        filename = Path(filename)
        if filename.exists():
            dt_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            new = Path(filename.parent, f"{filename.stem}_{dt_str}{filename.suffix}")
            print(f"File exists: {filename}, saving as '{new}' instead.")
            filename = new
        self.filename = str(filename)
        try:
            pagesize = getattr(pagesizes, pagesize)
        except AttributeError as exc:
            valid = [k for k, v in pagesizes.__dict__.items() if isinstance(v, tuple)]
            raise AttributeError(f"Invalid pagesize: {pagesize}. Valid are: {valid}") from exc
        if orientation.lower() not in ["landscape", "portrait"]:
            raise ValueError(f"orientation must be 'landscape' or 'portrait', not '{orientation}'")
        self.pagesize = (
            pagesizes.landscape(pagesize) if orientation.lower() == "landscape" else pagesize
        )

        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        if pagination_strategy.lower() not in ["rows_first", "columns_first"]:
            raise ValueError(
                f"Allowed pagination_strategy: 'rows_first' or 'columns_first', not '{pagination_strategy}'"
            )
        self.pagination_strategy = pagination_strategy.lower()
        self.min_width_factor = min_width_factor

        # Calculate usable page dimensions (accounting for margins)
        self.page_width = self.pagesize[0] - 0.8 * inch
        self.page_height = self.pagesize[1] - 0.8 * inch

        # Initialize document
        self.doc = SimpleDocTemplate(
            self.filename,
            pagesize=self.pagesize,
            rightMargin=0.4 * inch,
            leftMargin=0.4 * inch,
            topMargin=0.4 * inch,
            bottomMargin=0.4 * inch,
        )

        self.styles = getSampleStyleSheet()
        self.story = []

    def calculate_optimal_scaling(self, df):
        """
        Calculate optimal font size and column widths based on DataFrame dimensions.
        Uses min_width_factor to control column compression.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            dict: Scaling parameters including whether columns need splitting
        """
        num_cols = len(df.columns)
        if isinstance(self.min_width_factor, Sequence) and len(self.min_width_factor) != num_cols:
            raise ValueError(
                "When a list of min_width_factor is given, it must have the same "
                "number of columns as the dataframe."
            )

        # Calculate ideal column widths based on content
        ideal_col_widths = []
        for col in df.columns:
            # Check column header length
            header_len = len(str(col))

            # Check content lengths (use 90th percentile to avoid outliers)
            if not df[col].empty:
                content_lengths = df[col].astype(str).str.len()
                # Use 90th percentile to avoid extreme outliers affecting all columns
                typical_content_len = content_lengths.quantile(0.9)
                max_content_len = max(header_len, typical_content_len)
            else:
                max_content_len = header_len

            # Estimate required width (rough approximation)
            # This gives us the "ideal" width for unwrapped content
            ideal_width = max_content_len * 0.009 * self.min_font_size * inch
            # Set reasonable bounds
            ideal_width = max(0.3 * inch, min(ideal_width, 6.0 * inch))
            ideal_col_widths.append(ideal_width)

        # Calculate total ideal width
        total_ideal_width = sum(ideal_col_widths)

        # Apply minimum width factor to determine actual minimum widths
        if isinstance(self.min_width_factor, Sequence):
            min_col_widths = [
                ideal_col_widths[i] * self.min_width_factor[i] for i in range(num_cols)
            ]
        else:
            min_col_widths = [width * self.min_width_factor for width in ideal_col_widths]
        total_min_width = sum(min_col_widths)

        # Determine if we need to split columns
        needs_column_splitting = total_min_width > self.page_width

        if needs_column_splitting:
            # Columns need to be split across pages
            # Use minimum widths as they are - don't compress further
            final_col_widths = min_col_widths
            font_size = self.min_font_size  # Use smallest font for wide tables
        else:
            # All columns can fit on one page
            if total_ideal_width <= self.page_width:
                # Ideal widths fit perfectly
                final_col_widths = ideal_col_widths
                font_size = self.max_font_size
            else:
                # Need some compression, but not below minimum factor
                # Scale down proportionally, but respect minimum widths
                scale_factor = self.page_width / total_ideal_width

                if scale_factor >= self.min_width_factor:
                    # We can fit by scaling down, but not below minimum factor
                    final_col_widths = [width * scale_factor for width in ideal_col_widths]
                    font_size = max(self.min_font_size, int(self.max_font_size * scale_factor))
                else:
                    # Even with minimum factor, we'd need to split
                    final_col_widths = min_col_widths
                    needs_column_splitting = True
                    font_size = self.min_font_size

        # Calculate row height based on font size
        # This value is manually tweaked so that it matches the outcome of current table_style.
        # Only tested for font size 6. May wrap over page on larger font size.
        row_height = font_size * 1.5

        return {
            "font_size": font_size,
            "col_widths": final_col_widths,
            "row_height": row_height,
            "needs_column_splitting": needs_column_splitting,
            "ideal_col_widths": ideal_col_widths,
            "compression_ratio": (
                sum(final_col_widths) / sum(ideal_col_widths) if sum(ideal_col_widths) > 0 else 1.0
            ),
        }

    def calculate_max_rows_per_page(self, row_height):
        """
        Calculate maximum rows that can fit on a page given the fixed table position.

        Args:
            row_height (float): Height of each row in points

        Returns:
            int: Maximum number of rows per page
        """
        # Account for header row height
        header_height = row_height * 1.2  # Header typically slightly taller

        # Calculate how many data rows can fit (excluding header)
        available_for_data = self.page_height - header_height
        max_data_rows = int(available_for_data / row_height)

        # Ensure at least 1 row per page
        return max(1, max_data_rows)

    def wrap_text(self, text, max_width, font_size):
        """
        Wrap text to fit within specified width.

        Args:
            text (str): Text to wrap
            max_width (float): Maximum width in points
            font_size (int): Font size

        Returns:
            str: Wrapped text with line breaks
        """
        if not text or text == "nan":
            return ""

        text = str(text)
        # Rough estimate: each character takes about 0.6 * font_size points
        char_width = 0.4 * font_size
        chars_per_line = int(max_width / char_width)

        if chars_per_line <= 0:
            chars_per_line = 10  # Minimum

        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word)
            # Check if adding this word would exceed the line limit
            if current_length + word_length + len(current_line) > chars_per_line and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def create_table_style(self, font_size, num_cols, num_rows):
        """
        Create a table style with the specified font size.

        Args:
            font_size (int): Font size for the table
            num_cols (int): Number of columns
            num_rows (int): Number of rows

        Returns:
            TableStyle: Formatted table style
        """
        style = TableStyle(
            [
                # Header style
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), font_size),
                ("BOTTOMPADDING", (0, 0), (-1, -1), -4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                # Data style
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), font_size),
                # Grid
                ("GRID", (0, 0), (-1, -1), 0.1, colors.black),
                # Alternating row colors
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.beige, colors.lightgrey]),
            ]
        )

        return style

    def split_columns_for_pages(self, df, col_widths):
        """
        Split DataFrame columns into chunks that fit on page width.

        Args:
            df (pd.DataFrame): Input DataFrame
            col_widths (list): List of column widths

        Returns:
            list: List of tuples (column_indices, total_width)
        """
        column_groups = []
        current_group = []
        current_width = 0

        for i, width in enumerate(col_widths):
            if current_width + width <= self.page_width:
                current_group.append(i)
                current_width += width
            else:
                if current_group:  # Save current group if not empty
                    column_groups.append((current_group, current_width))
                current_group = [i]
                current_width = width

        # Add the last group
        if current_group:
            column_groups.append((current_group, current_width))

        return column_groups

    def split_dataframe_for_pages(self, df, max_rows_per_page=50):
        """
        Split DataFrame into chunks that fit on pages.

        Args:
            df (pd.DataFrame): Input DataFrame
            max_rows_per_page (int): Maximum rows per page

        Returns:
            list: List of DataFrame chunks
        """
        chunks = []
        for i in range(0, len(df), max_rows_per_page):
            chunk = df.iloc[i : i + max_rows_per_page].copy()
            chunks.append(chunk)
        return chunks

    def add_dataframe_to_pdf(self, df, title="", max_rows_per_page=50):
        """
        Add a DataFrame to the PDF with automatic scaling and multi-page support.

        Args:
            df (pd.DataFrame): Input DataFrame
            title (str): Title for the table
            max_rows_per_page (int): Maximum rows per page
        """
        # Calculate optimal scaling
        scaling_params = self.calculate_optimal_scaling(df)

        max_rows_per_page = self.calculate_max_rows_per_page(scaling_params["row_height"])

        needs_column_splitting = scaling_params["needs_column_splitting"]

        title = f"{title}: " if title else ""
        section_style = ParagraphStyle(
            "SectionTitle",
            parent=self.styles["Heading2"],
            fontSize=8,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
        if needs_column_splitting:
            # Split columns into groups that fit on pages
            column_groups = self.split_columns_for_pages(df, scaling_params["col_widths"])

            if self.pagination_strategy == "rows_first":
                total_pages = len(column_groups) * ((len(df) - 1) // max_rows_per_page + 1)
                # self.story.append(Spacer(1, 20))

                # Rows-first: For each column group, print all rows
                page_counter = 1
                for col_group_idx, (col_indices, group_width) in enumerate(column_groups):
                    # Create DataFrame subset for this column group
                    col_subset = df.iloc[:, col_indices]
                    col_widths_subset = [scaling_params["col_widths"][i] for i in col_indices]

                    # Split rows into pages
                    row_chunks = self.split_dataframe_for_pages(col_subset, max_rows_per_page)

                    for row_chunk_idx, row_chunk in enumerate(row_chunks):
                        # Add section header
                        section_title = (
                            f"{title}Columns {col_indices[0]+1}-{col_indices[-1]+1} "
                            f"(Page {page_counter} of {total_pages})"
                        )
                        self.story.append(Paragraph(section_title, section_style))

                        # Create and add table
                        self._create_table_section(
                            row_chunk,
                            col_widths_subset,
                            scaling_params,
                            page_counter,
                            total_pages,
                        )
                        page_counter += 1

                        # Add page break if not last
                        self.story.append(PageBreak())

            else:  # columns_first
                total_pages = ((len(df) - 1) // max_rows_per_page + 1) * len(column_groups)
                # self.story.append(Spacer(1, 20))

                # Columns-first: For each row group, print all column groups
                row_chunks = self.split_dataframe_for_pages(df, max_rows_per_page)
                page_counter = 1

                for row_chunk_idx, row_chunk in enumerate(row_chunks):
                    for col_group_idx, (col_indices, group_width) in enumerate(column_groups):
                        # Create DataFrame subset for this combination
                        table_subset = row_chunk.iloc[:, col_indices]
                        col_widths_subset = [scaling_params["col_widths"][i] for i in col_indices]

                        # Add section header
                        section_title = (
                            f"{title}Rows {row_chunk_idx*max_rows_per_page+1}-"
                            f"{min((row_chunk_idx+1)*max_rows_per_page, len(df))}, "
                            f"Columns {col_indices[0]+1}-{col_indices[-1]+1} "
                            f"(Page {page_counter} of {total_pages})"
                        )

                        self.story.append(Paragraph(section_title, section_style))

                        # Create and add table
                        self._create_table_section(
                            table_subset,
                            col_widths_subset,
                            scaling_params,
                            page_counter,
                            total_pages,
                        )
                        page_counter += 1

                        # Add page break if not last
                        self.story.append(PageBreak())

        else:
            # Table fits in width, use original logic
            total_pages = (len(df) - 1) // max_rows_per_page + 1
            # self.story.append(Spacer(1, 20))

            # Split DataFrame into page-sized chunks
            df_chunks = self.split_dataframe_for_pages(df, max_rows_per_page)

            for i, chunk in enumerate(df_chunks):
                # Add section header
                section_title = (
                    f"{title}Rows {i*max_rows_per_page+1}-"
                    f"{min((i+1)*max_rows_per_page, len(df))}, "
                    f"(Page {i} of {total_pages})"
                )

                self.story.append(Paragraph(section_title, section_style))
                self._create_table_section(
                    chunk,
                    scaling_params["col_widths"],
                    scaling_params,
                    i + 1,
                    len(df_chunks),
                )

                # Add page break if not last chunk
                self.story.append(PageBreak())

    def _create_table_section(self, df_chunk, col_widths, scaling_params, page_num, total_pages):
        """
        Create a table section for a DataFrame chunk.

        Args:
            df_chunk (pd.DataFrame): DataFrame chunk to display
            col_widths (list): Column widths for this chunk
            scaling_params (dict): Scaling parameters
            page_num (int): Current page number
            total_pages (int): Total number of pages
        """
        # Prepare data for table with text wrapping
        data = []

        # Add headers with wrapping
        headers = []
        for j, col in enumerate(df_chunk.columns):
            wrapped_header = self.wrap_text(str(col), col_widths[j], scaling_params["font_size"])
            headers.append(wrapped_header)
        data.append(headers)

        # Add data rows with wrapping
        for _, row in df_chunk.iterrows():
            wrapped_row = []
            for j, val in enumerate(row):
                wrapped_val = self.wrap_text(str(val), col_widths[j], scaling_params["font_size"])
                wrapped_row.append(wrapped_val)
            data.append(wrapped_row)

        # Create table with calculated column widths
        table = Table(data, colWidths=col_widths, repeatRows=1)

        # Apply style
        table_style = self.create_table_style(
            scaling_params["font_size"], len(df_chunk.columns), len(df_chunk)
        )
        table.setStyle(table_style)

        self.story.append(table)

    def generate_pdf(self):
        """Generate the final PDF document."""
        self.doc.build(self.story)
        print(f"PDF generated successfully: {self.filename}")


# Example usage with different compression factors
def main():
    # Create a wide sample DataFrame to demonstrate width handling
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Product_ID": [f"PROD_{i:04d}" for i in range(30)],
            "Product_Name": [f"Product Name {i} - Extended Description" for i in range(30)],
            "Category": np.random.choice(
                [
                    "Electronics & Gadgets",
                    "Clothing & Accessories",
                    "Home & Garden Supplies",
                    "Sports & Recreation",
                ],
                30,
            ),
            "Price": np.random.uniform(10, 500, 30).round(2),
            "Stock_Quantity": np.random.randint(0, 1000, 30),
            "Supplier": [f"Supplier Company {chr(65 + i % 26)} Limited" for i in range(30)],
            "Last_Updated": pd.date_range("2024-01-01", periods=30, freq="D"),
            "Description": [
                f"Detailed description for product {i} with various specifications and features"
                for i in range(30)
            ],
            "Description2": [
                f"Detailed description for product {i} with various specifications and features"
                for i in range(30)
            ],
            "Description3": [
                f"Detailed description for product {i} with various specifications and features"
                for i in range(30)
            ],
            "Manufacturer": [f"Manufacturer {chr(65 + i % 10)} Industries" for i in range(30)],
            "Country_Origin": np.random.choice(
                ["United States", "Germany", "Japan", "South Korea", "China"], 30
            ),
            "Warranty_Period": np.random.choice(
                ["1 Year Limited", "2 Years Full", "3 Years Extended", "6 Months Basic"], 30
            ),
            "Product_Weight": np.random.uniform(0.1, 50.0, 30).round(2),
            "Dimensions": [
                f"{np.random.randint(5, 50)}x{np.random.randint(5, 50)}x{np.random.randint(5, 50)} cm"
                for _ in range(30)
            ],
            "Color_Options": [f"Available in {np.random.randint(2, 8)} colors" for _ in range(30)],
            "Special_Features": [
                f"Feature Set {i}: Advanced technology with modern design" for i in range(30)
            ],
            "Availability": np.random.choice(
                ["In Stock", "Limited", "Out of Stock", "Discontinued"], 30
            ),
        }
    )

    # Test different compression factors
    compression_factors = [
        1,
        [1, 0.55, 0.55, 1, 1, 0.55, 1, 0.2, 0.25, 0.33, 0.55, 1, 0.55, 1, 1, 0.55, 0.55, 1],
    ]

    for i, factor in enumerate(compression_factors):
        for j, strategy in enumerate(["rows_first", "columns_first"]):
            print(f"\nTesting with min_width_factor = {factor}...")

            # Create PDF generator with specific compression factor
            pdf_generator = DataFrameToPDF(
                filename=f"wide_table_compression_case{i}_{strategy}.pdf",
                pagesize="A4",
                orientation="landscape",
                min_font_size=6,
                max_font_size=10,
                pagination_strategy=strategy,
                min_width_factor=factor,
            )

            # Add DataFrame to PDF
            pdf_generator.add_dataframe_to_pdf(
                df, title=f"Product Inventory - Min Width Factor {factor}", max_rows_per_page=50
            )

            # Generate PDF
            pdf_generator.generate_pdf()

    print("\nAll PDFs generated successfully!")
    print("Compare the following files to see the effect of different compression factors:")
    for i, factor in enumerate(compression_factors):
        print(f"- wide_table_compression_case{i}.pdf (min_width_factor = {factor})")

    print("\nKey differences:")
    print("- 0.50: Columns can be compressed to 50% of ideal width")
    print("- 0.33: Columns can be compressed to 33% of ideal width")
    print("- 0.25: Columns can be compressed to 25% of ideal width")
    print("- Lower values = more text wrapping, fewer page splits")
    print("- Higher values = less text wrapping, more page splits")


if __name__ == "__main__":
    pass
    # main()
