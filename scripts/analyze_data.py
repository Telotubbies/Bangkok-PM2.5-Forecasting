#!/usr/bin/env python3
"""
Analyze data directory structure and quality
"""
import sys
from pathlib import Path
import glob
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: pandas not installed. Please run: pip install pandas pyarrow")
    sys.exit(1)


def analyze_data_structure():
    """Analyze complete data directory structure"""
    print("="*70)
    print("DATA STRUCTURE ANALYSIS")
    print("="*70)
    
    data_dir = project_root / "data"
    
    # Check directories
    print("\n📁 Directory Structure:")
    for subdir in ['bronze', 'silver', 'stations', 'processed']:
        path = data_dir / subdir
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            files = len(list(path.rglob('*.parquet')))
            print(f"  ✅ {subdir:12s} - {size_mb:8.1f} MB - {files:5d} parquet files")
        else:
            print(f"  ❌ {subdir:12s} - MISSING")
    
    return data_dir


def analyze_air_quality(data_dir):
    """Analyze air quality data"""
    print("\n" + "="*70)
    print("AIR QUALITY DATA (PM2.5)")
    print("="*70)
    
    aq_dir = data_dir / "silver" / "openmeteo_airquality"
    if not aq_dir.exists():
        print("❌ Air quality data not found!")
        return None
    
    # Find all parquet files
    aq_files = list(aq_dir.rglob("*.parquet"))
    print(f"\n📊 Total files: {len(aq_files)}")
    
    if not aq_files:
        print("❌ No parquet files found!")
        return None
    
    # Sample first file
    try:
        sample = pd.read_parquet(aq_files[0])
        print(f"\n📋 Columns: {list(sample.columns)}")
        print(f"   Shape: {sample.shape}")
        
        # Check for PM2.5 data
        pm25_cols = [col for col in sample.columns if 'pm2' in col.lower() or 'pm_2' in col.lower()]
        print(f"   PM2.5 columns: {pm25_cols}")
        
        # Load all data to check date range
        print("\n⏳ Loading all air quality data...")
        all_data = []
        for f in aq_files[:100]:  # Sample first 100 files
            try:
                df = pd.read_parquet(f)
                all_data.append(df)
            except Exception as e:
                print(f"   ⚠️  Error reading {f.name}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            if 'date' in combined.columns:
                combined['date'] = pd.to_datetime(combined['date'])
                print(f"\n📅 Date Range:")
                print(f"   Start: {combined['date'].min()}")
                print(f"   End:   {combined['date'].max()}")
                print(f"   Days:  {(combined['date'].max() - combined['date'].min()).days}")
            
            if 'stationID' in combined.columns:
                print(f"\n🗺️  Stations: {combined['stationID'].nunique()} unique")
            
            # Check PM2.5 data quality
            if pm25_cols:
                pm25_col = pm25_cols[0]
                print(f"\n📈 PM2.5 Data Quality ({pm25_col}):")
                print(f"   Total records: {len(combined):,}")
                print(f"   Non-null: {combined[pm25_col].notna().sum():,} ({combined[pm25_col].notna().sum()/len(combined)*100:.1f}%)")
                print(f"   Null: {combined[pm25_col].isna().sum():,} ({combined[pm25_col].isna().sum()/len(combined)*100:.1f}%)")
                
                if combined[pm25_col].notna().any():
                    print(f"   Mean: {combined[pm25_col].mean():.2f} µg/m³")
                    print(f"   Std:  {combined[pm25_col].std():.2f} µg/m³")
                    print(f"   Min:  {combined[pm25_col].min():.2f} µg/m³")
                    print(f"   Max:  {combined[pm25_col].max():.2f} µg/m³")
            
            return combined
        
    except Exception as e:
        print(f"❌ Error analyzing air quality data: {e}")
        import traceback
        traceback.print_exc()
    
    return None


def analyze_weather(data_dir):
    """Analyze weather data"""
    print("\n" + "="*70)
    print("WEATHER DATA")
    print("="*70)
    
    weather_dir = data_dir / "silver" / "openmeteo_weather"
    if not weather_dir.exists():
        print("❌ Weather data not found!")
        return None
    
    weather_files = list(weather_dir.rglob("*.parquet"))
    print(f"\n📊 Total files: {len(weather_files)}")
    
    if not weather_files:
        print("❌ No parquet files found!")
        return None
    
    try:
        sample = pd.read_parquet(weather_files[0])
        print(f"\n📋 Columns: {list(sample.columns)}")
        print(f"   Shape: {sample.shape}")
        
        # Load sample data
        print("\n⏳ Loading sample weather data...")
        all_data = []
        for f in weather_files[:100]:
            try:
                df = pd.read_parquet(f)
                all_data.append(df)
            except:
                pass
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            
            if 'date' in combined.columns:
                combined['date'] = pd.to_datetime(combined['date'])
                print(f"\n📅 Date Range:")
                print(f"   Start: {combined['date'].min()}")
                print(f"   End:   {combined['date'].max()}")
            
            if 'stationID' in combined.columns:
                print(f"\n🗺️  Stations: {combined['stationID'].nunique()} unique")
            
            return combined
            
    except Exception as e:
        print(f"❌ Error analyzing weather data: {e}")
    
    return None


def analyze_stations(data_dir):
    """Analyze station metadata"""
    print("\n" + "="*70)
    print("STATION METADATA")
    print("="*70)
    
    stations_file = data_dir / "stations" / "bangkok_stations.parquet"
    if not stations_file.exists():
        print("❌ Station metadata not found!")
        return None
    
    try:
        stations = pd.read_parquet(stations_file)
        print(f"\n📊 Total stations: {len(stations)}")
        print(f"   Columns: {list(stations.columns)}")
        
        if 'lat' in stations.columns and 'lon' in stations.columns:
            print(f"\n🗺️  Geographic Coverage:")
            print(f"   Latitude:  {stations['lat'].min():.4f} to {stations['lat'].max():.4f}")
            print(f"   Longitude: {stations['lon'].min():.4f} to {stations['lon'].max():.4f}")
        
        if 'stationID' in stations.columns:
            print(f"\n🏷️  Station IDs:")
            print(f"   {stations['stationID'].tolist()[:10]}")
            if len(stations) > 10:
                print(f"   ... and {len(stations)-10} more")
        
        return stations
        
    except Exception as e:
        print(f"❌ Error analyzing stations: {e}")
    
    return None


def check_data_alignment(aq_data, weather_data, stations):
    """Check if data sources are aligned"""
    print("\n" + "="*70)
    print("DATA ALIGNMENT CHECK")
    print("="*70)
    
    issues = []
    
    # Check station alignment
    if stations is not None and 'stationID' in stations.columns:
        station_ids = set(stations['stationID'].unique())
        
        if aq_data is not None and 'stationID' in aq_data.columns:
            aq_stations = set(aq_data['stationID'].unique())
            missing_in_aq = station_ids - aq_stations
            extra_in_aq = aq_stations - station_ids
            
            print(f"\n🔍 Air Quality vs Stations:")
            print(f"   Stations in metadata: {len(station_ids)}")
            print(f"   Stations in AQ data: {len(aq_stations)}")
            print(f"   Missing in AQ: {len(missing_in_aq)}")
            print(f"   Extra in AQ: {len(extra_in_aq)}")
            
            if missing_in_aq:
                issues.append(f"Missing {len(missing_in_aq)} stations in air quality data")
            if extra_in_aq:
                issues.append(f"Extra {len(extra_in_aq)} stations in air quality data not in metadata")
        
        if weather_data is not None and 'stationID' in weather_data.columns:
            weather_stations = set(weather_data['stationID'].unique())
            missing_in_weather = station_ids - weather_stations
            
            print(f"\n🔍 Weather vs Stations:")
            print(f"   Stations in metadata: {len(station_ids)}")
            print(f"   Stations in weather data: {len(weather_stations)}")
            print(f"   Missing in weather: {len(missing_in_weather)}")
            
            if missing_in_weather:
                issues.append(f"Missing {len(missing_in_weather)} stations in weather data")
    
    # Check date alignment
    if aq_data is not None and weather_data is not None:
        if 'date' in aq_data.columns and 'date' in weather_data.columns:
            aq_dates = set(pd.to_datetime(aq_data['date']).dt.date)
            weather_dates = set(pd.to_datetime(weather_data['date']).dt.date)
            
            print(f"\n📅 Date Overlap:")
            print(f"   AQ dates: {len(aq_dates)}")
            print(f"   Weather dates: {len(weather_dates)}")
            print(f"   Common dates: {len(aq_dates & weather_dates)}")
            
            if len(aq_dates & weather_dates) == 0:
                issues.append("No overlapping dates between AQ and weather data")
    
    return issues


def generate_recommendations(issues):
    """Generate improvement recommendations"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if not issues:
        print("\n✅ No major issues found!")
    else:
        print("\n⚠️  Issues Found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    
    print("\n📋 Improvement Plan:")
    print("\n1. **Data Completeness**")
    print("   - Check for missing PM2.5 measurements")
    print("   - Fill gaps with interpolation or external sources")
    print("   - Verify all stations have data for training period")
    
    print("\n2. **Data Quality**")
    print("   - Remove outliers (PM2.5 > 500 µg/m³)")
    print("   - Handle missing values (forward fill, interpolation)")
    print("   - Validate weather data consistency")
    
    print("\n3. **Data Preparation for STC-HGAT**")
    print("   - Create unified dataset with all features")
    print("   - Normalize/standardize features")
    print("   - Create train/val/test splits (70/15/15)")
    print("   - Save processed data to data/processed/")
    
    print("\n4. **Feature Engineering**")
    print("   - Add temporal features (day of week, month, season)")
    print("   - Add spatial features (distance to city center)")
    print("   - Add lagged features (PM2.5 t-1, t-7)")
    print("   - Calculate rolling statistics")
    
    print("\n5. **DVC Pipeline Integration**")
    print("   - Track raw data with DVC")
    print("   - Create reproducible preprocessing pipeline")
    print("   - Version control processed datasets")


def main():
    """Main analysis function"""
    print("\n🔍 Bangkok PM2.5 Data Analysis\n")
    
    data_dir = analyze_data_structure()
    
    # Analyze each data source
    aq_data = analyze_air_quality(data_dir)
    weather_data = analyze_weather(data_dir)
    stations = analyze_stations(data_dir)
    
    # Check alignment
    issues = check_data_alignment(aq_data, weather_data, stations)
    
    # Generate recommendations
    generate_recommendations(issues)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
