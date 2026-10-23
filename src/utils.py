import pandas as pd

def summarize_detections(detections):
    """
    Summarizes detections using Peak Concurrent Object counting per frame
    to ensure precise industrial statistics.
    """
    if not detections:
        return pd.DataFrame(columns=["Detected Item", "Count", "Total Frame Occurrences"])

    df = pd.DataFrame(detections)
    
    # Calculate peak count per frame if frame_id exists
    if 'frame_id' in df.columns and not df.empty:
        frame_counts = df.groupby(['frame_id', 'class']).size().reset_index(name='single_frame_count')
        
        peak_df = frame_counts.groupby('class')['single_frame_count'].max().reset_index()
        peak_df.columns = ['Detected Item', 'Count']
        
        total_df = df.groupby('class').size().reset_index(name='Total Frame Occurrences')
        total_df.columns = ['Detected Item', 'Total Frame Occurrences']
        
        summary_df = pd.merge(peak_df, total_df, on='Detected Item')
    else:
        summary_df = df['class'].value_counts().reset_index()
        summary_df.columns = ['Detected Item', 'Count']
        summary_df['Total Frame Occurrences'] = summary_df['Count']

    return summary_df.sort_values(by="Count", ascending=False)