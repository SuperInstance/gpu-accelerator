//! Core types for gpu-accelerator

use serde::{Deserialize, Serialize};

/// Audio buffer for GPU processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBuffer {
    data: Vec<f32>,
    sample_rate: u32,
}

impl AudioBuffer {
    /// Create new audio buffer from slice
    pub fn from_slice(data: &[f32]) -> Self {
        Self {
            data: data.to_vec(),
            sample_rate: 16000, // Default 16kHz
        }
    }

    /// Create new audio buffer from vec
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data,
            sample_rate: 16000,
        }
    }

    /// Generate test audio data
    pub fn from_test_data() -> Self {
        Self::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5])
    }

    /// Get audio data
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get length in samples
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Valence: -1.0 (negative) to 1.0 (positive)
    pub valence: f32,

    /// Arousal: 0.0 (calm) to 1.0 (excited)
    pub arousal: f32,

    /// Dominance: 0.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
}

impl SentimentResult {
    pub fn new(valence: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            valence,
            arousal,
            dominance,
        }
    }

    /// Check if sentiment is positive
    pub fn is_positive(&self) -> bool {
        self.valence > 0.0
    }

    /// Check if sentiment is negative
    pub fn is_negative(&self) -> bool {
        self.valence < 0.0
    }
}

/// VAD (Voice Activity Detection) result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VADResult {
    /// Speech probability: 0.0 to 1.0
    pub speech_probability: f32,

    /// Energy level: 0.0 to 1.0
    pub energy_level: f32,

    /// Silence detected
    pub silence_detected: bool,
}

impl VADResult {
    pub fn new(speech_probability: f32, energy_level: f32) -> Self {
        Self {
            speech_probability,
            energy_level,
            silence_detected: speech_probability < 0.3,
        }
    }

    /// Check if speech is detected
    pub fn has_speech(&self) -> bool {
        self.speech_probability > 0.5
    }

    /// Check if silence is detected
    pub fn is_silence(&self) -> bool {
        self.silence_detected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer() {
        let audio = AudioBuffer::from_slice(&[0.1, 0.2, 0.3]);
        assert_eq!(audio.len(), 3);
        assert!(!audio.is_empty());
        assert_eq!(audio.sample_rate(), 16000);
    }

    #[test]
    fn test_sentiment_result() {
        let sentiment = SentimentResult::new(0.8, 0.6, 0.4);
        assert!(sentiment.is_positive());
        assert!(!sentiment.is_negative());
    }

    #[test]
    fn test_vad_result() {
        let vad = VADResult::new(0.9, 0.7);
        assert!(vad.has_speech());
        assert!(!vad.is_silence());

        let vad_silent = VADResult::new(0.1, 0.1);
        assert!(!vad_silent.has_speech());
        assert!(vad_silent.is_silence());
    }
}
