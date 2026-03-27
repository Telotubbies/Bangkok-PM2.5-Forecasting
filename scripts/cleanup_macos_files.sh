#!/bin/bash
# Clean up macOS metadata files from data directory

echo "=========================================="
echo "Cleaning macOS Metadata Files"
echo "=========================================="

echo ""
echo "🔍 Finding ._ files..."
count_underscore=$(find data/ -name "._*" -type f 2>/dev/null | wc -l)
echo "   Found: $count_underscore files"

if [ "$count_underscore" -gt 0 ]; then
    echo "   Removing ._ files..."
    find data/ -name "._*" -type f -delete
    echo "   ✅ Removed $count_underscore ._ files"
else
    echo "   ✅ No ._ files found"
fi

echo ""
echo "🔍 Finding .DS_Store files..."
count_dsstore=$(find data/ -name ".DS_Store" -type f 2>/dev/null | wc -l)
echo "   Found: $count_dsstore files"

if [ "$count_dsstore" -gt 0 ]; then
    echo "   Removing .DS_Store files..."
    find data/ -name ".DS_Store" -type f -delete
    echo "   ✅ Removed $count_dsstore .DS_Store files"
else
    echo "   ✅ No .DS_Store files found"
fi

echo ""
echo "🔍 Checking for other macOS metadata..."
count_appledb=$(find data/ -name ".AppleDB" -o -name ".AppleDesktop" -o -name ".AppleDouble" 2>/dev/null | wc -l)
if [ "$count_appledb" -gt 0 ]; then
    echo "   Found $count_appledb Apple metadata files"
    find data/ \( -name ".AppleDB" -o -name ".AppleDesktop" -o -name ".AppleDouble" \) -exec rm -rf {} + 2>/dev/null
    echo "   ✅ Removed Apple metadata"
else
    echo "   ✅ No Apple metadata found"
fi

echo ""
echo "=========================================="
echo "✅ Cleanup complete!"
echo ""
echo "Verifying data integrity..."
parquet_count=$(find data/ -name "*.parquet" -type f 2>/dev/null | wc -l)
echo "   Valid parquet files: $parquet_count"
echo "=========================================="
