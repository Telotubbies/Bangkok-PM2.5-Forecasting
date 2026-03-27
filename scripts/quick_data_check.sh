#!/bin/bash
# Quick data structure check

echo "=========================================="
echo "Bangkok PM2.5 Data - Quick Check"
echo "=========================================="

echo ""
echo "📁 Data Directory Sizes:"
du -sh data/*/ 2>/dev/null | sort -h

echo ""
echo "📊 Air Quality Data Years:"
find data/silver/openmeteo_airquality -type d -name "year=*" | sort | sed 's/.*year=/  - /'

echo ""
echo "🌤️  Weather Data Years:"
find data/silver/openmeteo_weather -type d -name "year=*" | sort | sed 's/.*year=/  - /'

echo ""
echo "📈 File Counts:"
echo "  Air Quality: $(find data/silver/openmeteo_airquality -name '*.parquet' 2>/dev/null | wc -l) parquet files"
echo "  Weather:     $(find data/silver/openmeteo_weather -name '*.parquet' 2>/dev/null | wc -l) parquet files"
echo "  Stations:    $(ls data/stations/*.parquet 2>/dev/null | wc -l) file(s)"

echo ""
echo "⚠️  Missing Years Check:"
for year in 2021 2022; do
    if [ ! -d "data/silver/openmeteo_airquality/year=$year" ]; then
        echo "  ❌ Air Quality $year - MISSING"
    else
        count=$(find data/silver/openmeteo_airquality/year=$year -name '*.parquet' 2>/dev/null | wc -l)
        if [ "$count" -eq 0 ]; then
            echo "  ❌ Air Quality $year - Directory exists but NO FILES"
        else
            echo "  ✅ Air Quality $year - $count files"
        fi
    fi
done

echo ""
echo "📂 Processed Data:"
if [ -d "data/processed" ]; then
    if [ "$(ls -A data/processed 2>/dev/null)" ]; then
        echo "  ✅ Exists with files:"
        ls -lh data/processed/ 2>/dev/null | tail -n +2 | awk '{print "     " $9 " (" $5 ")"}'
    else
        echo "  ⚠️  Directory exists but EMPTY"
    fi
else
    echo "  ❌ NOT CREATED YET"
fi

echo ""
echo "=========================================="
echo "✅ Quick check complete!"
echo ""
echo "Next steps:"
echo "  1. Review DATA_IMPROVEMENT_PLAN.md"
echo "  2. Run: python scripts/analyze_data.py (requires pandas)"
echo "  3. Address missing 2021-2022 data"
echo "=========================================="
