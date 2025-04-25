import speech_recognition as sr
from transformers import pipeline

class AudioCapture:
    def __init__(self):
        """Initialize speech recognition"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    
    def capture_audio(self):
        """Capture audio and return as text"""
        with self.microphone as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening... (speak now)")
            audio = self.recognizer.listen(source, timeout=30, phrase_time_limit=30)
        
        try:
            return self.recognizer.recognize_google(audio)
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from transformers import pipeline
class TextCleaner:
    def __init__(self, language='english'):
        """
        Initialize the TextCleaner with language-specific processing.
        Args:
            language: Language for processing (default: 'english')
        """
        self.stop_words = set(stopwords.words(language))
        
        # Ensure 'if', 'else', 'for', 'while' are NOT removed
        self.keep_words = {'if', 'else', 'for', 'while'}
        self.stop_words -= self.keep_words  # Exclude them from stopwords
        
        self.punctuation = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

        # Load Transformer model for grammar correction
        self.corrector = pipeline("text2text-generation", model="t5-small")

        # Common suffixes to remove
        self.suffixes = ['ing', 'tor', 'ment', 'tion', 'sion', 'ance', 'ence']
    
    def _is_meaningful(self, word):
        """Check if a word exists in WordNet (basic meaningful check)"""
        return bool(wn.synsets(word))

    def _stem_word(self, word):
        """Reduce word to its base form"""
        # Try lemmatizing first
        lemma = self.lemmatizer.lemmatize(word)
        if lemma != word:
            return lemma
            
        # Handle common suffixes if lemmatizer didn't work
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                base = word[:-len(suffix)]
                if self._is_meaningful(base):
                    return base
        
        return word

    def correct_text(self, text):
        """Use Transformer-based model to correct grammar & spelling."""
        corrected_text = self.corrector(f"{text}", max_length=100)[0]['generated_text']
        print("corejenvkejrfvkie:  ",corrected_text)
        return corrected_text

    def clean_text(self, text):
        """
        Clean the input text by:
        1. Correcting grammar & spelling using a Transformer model (T5)
        2. Converting to lowercase
        3. Removing special symbols/numbers
        4. Removing stop words (except 'if', 'else', 'for', 'while')
        5. Stemming words (removing common suffixes)
        6. Checking if words are meaningful
        Args:
            text: Input text to clean
            
        Returns:
            str: Cleaned, meaningful sentence
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        # Step 1: Correct grammar & spelling using Transformer
        text = self.correct_text(text)

        # Step 2: Convert to lowercase
        text = text.lower()

        # Step 3: Remove special symbols and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Step 4: Tokenize the text
        words = word_tokenize(text)

        # Step 5: Process each word
        cleaned_words = []
        for word in words:
            if word in self.stop_words or word in self.punctuation:
                continue  # Skip stopwords and punctuation
            
            # Step 6: Stem the word
            stemmed = self._stem_word(word)
            
            # Step 7: Check if meaningful
            if self._is_meaningful(stemmed) or word in self.keep_words:
                cleaned_words.append(stemmed)

        # Step 8: Join back into a single string
        return ' '.join(cleaned_words)


import os
import re
import time
import cv2
import numpy as np
from PIL import Image

class HandSignVideoDisplayer:
    def __init__(self, image_folder="hand_signs", display_width=800, display_height=600):
        """
        Initialize the video-style displayer
        
        Args:
            image_folder: Path to folder containing hand sign images/videos
            display_width: Width of output display window
            display_height: Height of output display window
        """
        self.image_folder = image_folder
        self.display_size = (display_width, display_height)
        self.letter_images = self._load_letter_images()
        self.word_videos = self._load_word_videos()
        self.fps = 24  # Frames per second for smooth playback
        self.word_duration = 5  # Seconds to display each word
        
    def _load_letter_images(self):
        """Load all letter images from the folder"""
        images = {}
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            img_path = os.path.join(self.image_folder, f"{letter}.jpg")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                # Resize to consistent size while maintaining aspect ratio
                img.thumbnail((200, 200))
                images[letter] = img
        return images
    
    def _load_word_videos(self):
        """Load all word videos from the folder"""
        videos = {}
        for filename in os.listdir(self.image_folder):
            if filename.endswith(".mp4"):
                word = filename[:-4].lower()  # Remove .mp4 and convert to lowercase
                videos[word] = os.path.join(self.image_folder, filename)
        return videos
    
    def _find_video_match(self, word):
        """Find exact video match for a word (no singular/plural conversion)"""
        word_lower = word.lower()
        return self.word_videos.get(word_lower)
    
    def _create_letter_composite(self, letters):
        """Create a composite image of all letters side by side"""
        images = [np.array(self.letter_images[letter]) for letter in letters]
        
        # Calculate total width and max height
        total_width = sum(img.shape[1] for img in images)
        max_height = max(img.shape[0] for img in images)
        
        # Create blank composite image
        composite = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        
        # Paste all images side by side
        x_offset = 0
        for img in images:
            h, w = img.shape[:2]
            composite[:h, x_offset:x_offset+w] = img
            x_offset += w
        
        return composite
    
    def _create_word_video(self, word):
        """Create a video frame sequence for a word"""
        frames = []
        
        # Try to find exact video match
        video_path = self._find_video_match(word)
        if video_path:
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.display_size)
                frames.append(frame)
            cap.release()
            
            # If video is shorter than desired duration, pad with last frame
            while len(frames) < self.fps * self.word_duration:
                frames.append(frames[-1])
        else:
            # Create side-by-side letter display
            letters = [c for c in word.lower() if c in self.letter_images]
            if not letters:
                return None
                
            # Create composite image of all letters
            composite = self._create_letter_composite(letters)
            
            # Resize to fit display while maintaining aspect ratio
            h, w = composite.shape[:2]
            scale = min(self.display_size[0]/w, self.display_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            composite = cv2.resize(composite, (new_w, new_h))
            
            # Center the image on a black background
            frame = np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8)
            y_offset = (self.display_size[1] - new_h) // 2
            x_offset = (self.display_size[0] - new_w) // 2
            frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = composite
            
            # Add word label
            cv2.putText(frame, word.upper(), 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            # Create frames for the full duration
            for _ in range(self.fps * self.word_duration):
                frames.append(frame.copy())
        
        return frames
    
    def display_text(self, text):
        """Display text as a continuous video stream"""
        words = re.findall(r"\b[\w'-]+\b", text.lower())
        all_frames = []
        
        # Generate frames for all words
        for word in words:
            print(f"Processing word: {word}")
            frames = self._create_word_video(word)
            if frames:
                all_frames.extend(frames)
                # Add brief blank frame between words
                all_frames.append(np.zeros((self.display_size[1], self.display_size[0], 3), dtype=np.uint8))
        
        if not all_frames:
            print("No content to display")
            return
        
        # Display the video stream
        cv2.namedWindow("Sign Language Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language Video", *self.display_size)
        
        for frame in all_frames:
            cv2.imshow("Sign Language Video", frame)
            if cv2.waitKey(int(1000/self.fps)) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    audio_capture = AudioCapture()
    recorded_text = audio_capture.capture_audio()

    if recorded_text:
        print(f"\nRecorded Text: {recorded_text}")
    else:
        print("No valid text was captured.")

    cleaner = TextCleaner()
    # Clean the text
    cleaned_text = cleaner.clean_text(recorded_text)
    
    print("\nCleaned Text: ",cleaned_text)
 

    displayer = HandSignVideoDisplayer(
        image_folder="C:/Users/charu/OneDrive/Documents/dhanya details/dhnayashree_mini_ptoject_code/hand_signs",# add your folder containg the images and video sources.
        display_width=800,
        display_height=600
    )
    
    displayer.display_text(cleaned_text)
