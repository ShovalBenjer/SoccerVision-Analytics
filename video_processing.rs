//! # Module `video_processing.rs`
//!
//! This module is responsible for handling video input, frame extraction, and basic
//! video quality assessment. It leverages the `ffmpeg-next` crate for video decoding
//! and the `image` crate for image manipulation and analysis.
//!
//! ## Functionality:
//!
//! 1.  **Frame Extraction:** Extracts frames from a video file at a specified frame rate.
//! 2.  **Video Quality Assessment:** Provides a basic score representing the video quality
//!     based on resolution, estimated sharpness, and estimated noise.
//!
//! ## Dependencies:
//!
//! *   `ffmpeg-next`: For video decoding and demuxing. Requires FFmpeg libraries to be installed
//!     on the system.
//! *   `image`: For image processing, format conversion, and basic image analysis.
//!
//! ## Usage:
//!
//! ```rust
//! use video_analytics::video_processing; // Assuming your crate is named video_analytics
//!
//! fn main() -> Result<(), String> {
//!     let video_path = "path/to/your/video.mp4"; // Replace with your video file path
//!     let frame_rate = 30.0; // Frames per second to extract
//!
//!     let frames_result = video_processing::extract_frames_from_video(video_path, frame_rate);
//!
//!     match frames_result {
//!         Ok(frames) => {
//!             println!("Successfully extracted {} frames.", frames.len());
//!             // Process the extracted frames
//!             for frame_data in &frames {
//!                 println!("Frame index: {}, Timestamp: {:.2}", frame_data.frame_index, frame_data.timestamp);
//!                 // Access frame_data.image (DynamicImage) for further processing
//!             }
//!         }
//!         Err(err) => {
//!             eprintln!("Error extracting frames: {}", err);
//!         }
//!     }
//!
//!     let quality_score = video_processing::calculate_video_quality_score(&frames_result.unwrap_or_default());
//!     println!("Video Quality Score: {:.2}", quality_score);
//!
//!     Ok(())
//! }
//! ```
//!
//! **Note:** This module provides basic functionality and serves as a starting point.
//! For production use, consider more robust error handling, advanced quality metrics,
//! and optimization for performance.

use ffmpeg_next as ffmpeg;
use image::{DynamicImage, ImageBuffer, Rgb, ImageError, imageops, Luma};
use std::path::Path;
use std::cmp;

/// Represents data for a single video frame.
#[derive(Debug)]
pub struct FrameData {
    /// The index of the frame in the video sequence (0-based).
    pub frame_index: usize,
    /// The image data of the frame as a `DynamicImage`.
    pub image: DynamicImage,
    /// The timestamp of the frame in seconds from the start of the video.
    pub timestamp: f64,
    // You can add more metadata here if needed, such as frame type, etc.
}

/// Extracts frames from a video file at a specified frame rate.
///
/// This function uses `ffmpeg-next` to decode the video and extract frames.
/// It supports common video formats that FFmpeg can handle.
///
/// # Arguments
///
/// *   `video_path`: The path to the video file.
/// *   `frame_rate`: The desired frame rate (frames per second) for extraction.
///                  If set to 0.0, it extracts frames at the video's native frame rate.
///
/// # Returns
///
/// *   `Result<Vec<FrameData>, String>`: On success, returns a `Vec` of `FrameData`
///     containing the extracted frames. On failure, returns an `Err` with an error message.
///
/// # Errors
///
/// This function can return an `Err` in the following cases:
///
/// *   If the video file does not exist.
/// *   If there are issues opening or decoding the video file using FFmpeg.
/// *   If no video stream is found in the input file.
/// *   If there are errors during image buffer creation.
pub fn extract_frames_from_video(video_path: &str, frame_rate: f64) -> Result<Vec<FrameData>, String> {
    let path = Path::new(video_path);
    if !path.exists() {
        return Err(format!("Video file not found: {}", video_path));
    }

    let mut frames = Vec::new();
    let mut ictx = ffmpeg::format::input(&path).map_err(|e| format!("FFmpeg input error: {:?}", e))?;

    let stream = ictx.streams().best(ffmpeg::media::Type::Video).ok_or("No video stream found")?;
    let stream_index = stream.index();

    let mut decoder = stream.codec().decoder().video().map_err(|e| format!("FFmpeg decoder error: {:?}", e))?;
    decoder.set_format(ffmpeg::format::Pixel::RGB24); // Force RGB24 pixel format for image crate compatibility

    let frame_rate_rational = stream.avg_frame_rate(); // Get the video's average frame rate as a rational number
    let video_frame_rate = frame_rate_rational.numerator() as f64 / frame_rate_rational.denominator() as f64;
    let frame_interval = if frame_rate > 0.0 && video_frame_rate > 0.0 { video_frame_rate / frame_rate } else { 1.0 }; // Calculate frame interval for desired frame rate

    let mut frame_no = 0;
    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_index {
            let mut decoded_frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::RGB24, decoder.width(), decoder.height());
            decoder.decode(&packet, &mut decoded_frame).map_err(|e| format!("FFmpeg decode error: {:?}", e))?;

            if decoded_frame.is_complete() && (frame_no as f64 % frame_interval) < 1.0 {
                let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(decoder.width(), decoder.height(), decoded_frame.data()[0].to_vec())
                        .ok_or("Failed to create ImageBuffer")?;

                let timestamp = decoded_frame.pts().map_or(0.0, |pts| pts as f64 * stream.time_base().numerator() as f64 / stream.time_base().denominator() as f64);

                let frame_data = FrameData {
                    frame_index: frame_no as usize,
                    image: DynamicImage::ImageRgb8(image_buffer),
                    timestamp, // Store frame timestamp
                };
                frames.push
