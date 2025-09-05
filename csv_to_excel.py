#!/usr/bin/env python3
"""
CSV to Excel Converter
Converts CSV files to Excel format with basic formatting
"""

import csv
import sys
from pathlib import Path

def csv_to_excel_manual(csv_file, excel_file):
    """Convert CSV to Excel using manual XML generation"""
    
    try:
        # Read CSV data
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        if not rows:
            print("No data found in CSV file")
            return False
        
        # Create Excel XML content
        xml_content = '''<?xml version="1.0"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
 <Worksheet ss:Name="NCCL_Results">
  <Table>
'''
        
        # Add header row with styling
        header_row = rows[0]
        xml_content += '   <Row>\n'
        for cell in header_row:
            xml_content += f'    <Cell><Data ss:Type="String">{cell}</Data></Cell>\n'
        xml_content += '   </Row>\n'
        
        # Add data rows
        for row in rows[1:]:
            xml_content += '   <Row>\n'
            for i, cell in enumerate(row):
                # Try to determine if it's a number
                try:
                    float(cell)
                    xml_content += f'    <Cell><Data ss:Type="Number">{cell}</Data></Cell>\n'
                except ValueError:
                    xml_content += f'    <Cell><Data ss:Type="String">{cell}</Data></Cell>\n'
            xml_content += '   </Row>\n'
        
        xml_content += '''  </Table>
 </Worksheet>
</Workbook>'''
        
        # Write Excel file
        with open(excel_file, 'w') as f:
            f.write(xml_content)
        
        print(f"Successfully converted {csv_file} to {excel_file}")
        return True
        
    except Exception as e:
        print(f"Error converting to Excel: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python csv_to_excel.py <csv_file>")
        print("Example: python csv_to_excel.py nccl_results.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    excel_file = Path(csv_file).stem + ".xls"
    
    if not Path(csv_file).exists():
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    
    print(f"Converting {csv_file} to Excel format...")
    
    if csv_to_excel_manual(csv_file, excel_file):
        print(f"Excel file created: {excel_file}")
        print("You can open this file in Excel, LibreOffice Calc, or Google Sheets")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()