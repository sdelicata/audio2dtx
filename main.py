import sys
import os
import logging
from audio_to_chart import AudioToChart

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for audio to DTX conversion"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_audio_file>")
        print("Example: python main.py song.mp3")
        sys.exit(1)

    input_filename = sys.argv[1]
    input_audio = f"/app/input/{input_filename}"
    output_dir = "/app/output"

    # Validate input file
    if not os.path.exists(input_audio):
        logger.error(f"File not found: {input_audio}")
        sys.exit(1)

    # Validate audio file format
    valid_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    file_ext = os.path.splitext(input_filename)[1].lower()
    if file_ext not in valid_formats:
        logger.error(f"Unsupported audio format: {file_ext}")
        logger.error(f"Supported formats: {', '.join(valid_formats)}")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info(f"Starting processing: {input_filename}")
        
        # Create chart converter instance
        chart = AudioToChart(input_audio)
        
        # Extract beats from audio
        logger.info("Extracting beats and creating chart...")
        chart.extract_beats()
        
        # Create DTX chart
        chart.create_chart()
        
        # Export complete simfile
        chart.export(output_dir)
        
        logger.info(f"✅ Successfully converted {input_filename} to DTXMania simfile")
        logger.info(f"📁 Output saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Error processing audio file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
