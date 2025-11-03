"""
Export Manager Module
Handles export to CSV, Excel, Parquet, and HTML formats
"""

import io
from datetime import datetime
from typing import Optional
import pandas as pd
import streamlit as st


class ExportManager:
    """Manages export of classified data to multiple formats"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize export manager
        Args:
            df: Classified transcripts dataframe
        """
        self.df = df
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_csv(self) -> bytes:
        """
        Export to CSV format
        Returns:
            CSV file bytes
        """
        return self.df.to_csv(index=False).encode('utf-8')
    
    def export_excel(self) -> bytes:
        """
        Export to Excel format with formatting
        Returns:
            Excel file bytes
        """
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='Classifications', index=False)
            
            # Get worksheet
            worksheet = writer.sheets['Classifications']
            
            # Auto-adjust column widths (max 50 characters)
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output.getvalue()
    
    def export_parquet(self) -> bytes:
        """
        Export to Parquet format (compressed columnar)
        Returns:
            Parquet file bytes
        """
        output = io.BytesIO()
        self.df.to_parquet(output, engine='pyarrow', compression='snappy', index=False)
        return output.getvalue()
    
    def export_html(self) -> str:
        """
        Export to HTML report format
        Returns:
            HTML string
        """
        # Calculate summary stats
        total = len(self.df)
        avg_conf = self.df['confidence'].mean()
        unique_cats = self.df['category'].nunique()
        unique_subcats = self.df['subcategory'].nunique()
        
        # Category distribution
        cat_dist = self.df['category'].value_counts().head(10)
        
        # Agent performance (if available)
        agent_html = ""
        if 'agent_name' in self.df.columns:
            agent_stats = self.df.groupby('agent_name').agg({
                'transcript_id': 'count',
                'confidence': 'mean',
                'category': 'nunique'
            }).reset_index()
            agent_stats.columns = ['Agent', 'Total Calls', 'Avg Confidence', 'Unique Categories']
            agent_stats = agent_stats.sort_values('Total Calls', ascending=False).head(10)
            
            agent_html = f"""
            <h2>👥 Top 10 Agents by Call Volume</h2>
            {agent_stats.to_html(index=False, classes='data-table')}
            """
        
        # Top issues
        top_issues = self.df.groupby(['category', 'subcategory']).size().reset_index(name='count')
        top_issues = top_issues.sort_values('count', ascending=False).head(10)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentPulse AI - Classification Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px 20px;
            color: #2d3748;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        h2 {{
            font-size: 1.8rem;
            margin: 40px 0 20px 0;
            color: #2d3748;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .data-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .data-table tbody tr:hover {{
            background: #f7fafc;
        }}
        
        .data-table tbody tr:nth-child(even) {{
            background: #f9fafb;
        }}
        
        .footer {{
            background: #f7fafc;
            padding: 20px 40px;
            text-align: center;
            color: #718096;
            font-size: 0.9rem;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 AgentPulse AI</h1>
            <p>Classification Report - Generated {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
        </div>
        
        <div class="content">
            <h2>📊 Summary Statistics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{total:,}</div>
                    <div class="metric-label">Total Transcripts</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_conf:.3f}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{unique_cats}</div>
                    <div class="metric-label">Unique Categories</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{unique_subcats}</div>
                    <div class="metric-label">Unique Subcategories</div>
                </div>
            </div>
            
            <h2>📈 Top 10 Categories</h2>
            {cat_dist.to_frame('count').to_html(classes='data-table')}
            
            {agent_html}
            
            <h2>🔍 Top 10 Issues</h2>
            {top_issues.to_html(index=False, classes='data-table')}
        </div>
        
        <div class="footer">
            <p>© 2025 AgentPulse AI - Enterprise QA & Coaching Platform</p>
            <p>Report generated using CCRE (Context Clustered Rule Engine) - 96% accuracy</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_template


def render_export_interface(df: pd.DataFrame):
    """
    Render export interface in Streamlit
    Args:
        df: Classified transcripts dataframe
    """
    st.header("📤 Export Manager")
    
    st.info(f"📊 Ready to export {len(df):,} classified transcripts")
    
    # Format selection
    st.subheader("📁 Select Export Formats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_csv = st.checkbox("📄 CSV", value=True, help="Comma-separated values (universal)")
        export_excel = st.checkbox("📊 Excel (XLSX)", help="Excel workbook with formatting")
    
    with col2:
        export_parquet = st.checkbox("🗜️ Parquet", help="Compressed columnar format (10-20x smaller)")
        export_html = st.checkbox("🌐 HTML Report", help="Styled HTML report for viewing")
    
    if not any([export_csv, export_excel, export_parquet, export_html]):
        st.warning("⚠️ Please select at least one export format")
        return
    
    st.markdown("---")
    
    # Generate exports
    if st.button("🚀 Generate Exports", type="primary", use_container_width=True):
        
        manager = ExportManager(df)
        timestamp = manager.timestamp
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        exports = []
        total_steps = sum([export_csv, export_excel, export_parquet, export_html])
        current_step = 0
        
        # CSV Export
        if export_csv:
            status_text.text("Generating CSV...")
            csv_data = manager.export_csv()
            exports.append(("CSV", f"agentpulse_classifications_{timestamp}.csv", csv_data, "text/csv"))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # Excel Export
        if export_excel:
            status_text.text("Generating Excel...")
            excel_data = manager.export_excel()
            exports.append(("Excel", f"agentpulse_classifications_{timestamp}.xlsx", excel_data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # Parquet Export
        if export_parquet:
            status_text.text("Generating Parquet...")
            parquet_data = manager.export_parquet()
            exports.append(("Parquet", f"agentpulse_classifications_{timestamp}.parquet", parquet_data, "application/octet-stream"))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # HTML Export
        if export_html:
            status_text.text("Generating HTML Report...")
            html_data = manager.export_html()
            exports.append(("HTML", f"agentpulse_report_{timestamp}.html", html_data.encode('utf-8'), "text/html"))
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        status_text.text("✅ Export generation complete!")
        
        st.markdown("---")
        
        # Display download buttons
        st.subheader("⬇️ Download Files")
        
        cols = st.columns(len(exports))
        
        for i, (format_name, filename, data, mime) in enumerate(exports):
            with cols[i]:
                st.download_button(
                    label=f"📥 {format_name}",
                    data=data,
                    file_name=filename,
                    mime=mime,
                    use_container_width=True
                )
                
                # Show file size
                size_mb = len(data) / (1024 * 1024)
                st.caption(f"Size: {size_mb:.2f} MB")
        
        st.success(f"✅ Generated {len(exports)} export file(s) successfully!")
        
        # Export tips
        with st.expander("💡 Export Format Tips"):
            st.markdown("""
            **CSV**: 
            - Universal format, works everywhere
            - Best for sharing with non-technical users
            - Larger file size
            
            **Excel (XLSX)**:
            - Formatted with auto-adjusted columns
            - Easy to view and analyze in Excel/Google Sheets
            - Slightly slower generation
            
            **Parquet**:
            - 10-20x smaller than CSV
            - Fast read/write operations
            - Ideal for data warehouses (Snowflake, BigQuery, Redshift)
            - Preserves data types
            
            **HTML Report**:
            - Styled, professional report
            - No Excel needed - view in any browser
            - Great for emailing to stakeholders
            - Includes summary statistics and visualizations
            """)
