"""
Music Theory Tool - A lightweight tool for music theory knowledge and assistance.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MusicTheoryTool:
    """
    A lightweight music theory tool that provides information about scales, chords,
    intervals, and other music theory concepts without requiring heavy dependencies.
    """
    
    def __init__(self):
        """Initialize the Music Theory Tool."""
        self.name = "MusicTheoryTool"
        self.description = "Provides music theory knowledge including scales, chords, intervals, and key signatures"
        self.logger = logger
        
        # Define major scales
        self.major_scales = {
            'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
            'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
            'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
            'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
            'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
            'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
            'F#': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
            'C#': ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'],
            'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
            'Bb': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
            'Eb': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
            'Ab': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
            'Db': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
            'Gb': ['Gb', 'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'F'],
            'Cb': ['Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb']
        }
        
        # Define basic chord types
        self.chord_patterns = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'dominant7': [0, 4, 7, 10],
            'diminished7': [0, 3, 6, 9]
        }
        
        # Circle of fifths
        self.circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F']
        
    def execute(self, query: str) -> str:
        """
        Execute a music theory query.
        
        Args:
            query: The user's query about music theory
            
        Returns:
            A response with music theory information
        """
        try:
            query_lower = query.lower()
            
            # Scale queries
            if 'scale' in query_lower:
                return self._handle_scale_query(query)
            
            # Chord queries
            elif 'chord' in query_lower:
                return self._handle_chord_query(query)
                
            # Interval queries
            elif 'interval' in query_lower:
                return self._handle_interval_query(query)
                
            # Key signature queries
            elif 'key' in query_lower and ('signature' in query_lower or 'sharp' in query_lower or 'flat' in query_lower):
                return self._handle_key_signature_query(query)
                
            # Circle of fifths
            elif 'circle' in query_lower and 'fifth' in query_lower:
                return self._explain_circle_of_fifths()
                
            # General music theory
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            self.logger.error(f"Error in music theory query: {e}")
            return f"I encountered an error processing your music theory question: {str(e)}"
    
    def _handle_scale_query(self, query: str) -> str:
        """Handle queries about musical scales."""
        query_lower = query.lower()
        
        # Extract key name
        for key in self.major_scales.keys():
            if key.lower() in query_lower:
                if 'major' in query_lower or 'scale' in query_lower:
                    notes = self.major_scales[key]
                    return f"The {key} major scale contains the notes: {', '.join(notes)}"
                    
        # General scale information
        if any(word in query_lower for word in ['what', 'explain', 'tell']):
            return """A musical scale is a sequence of musical notes in ascending or descending order. The most common scale is the major scale, which follows the pattern:

**Major Scale Pattern**: Whole-Whole-Half-Whole-Whole-Whole-Half steps

For example, C major scale: C-D-E-F-G-A-B
- C to D: Whole step
- D to E: Whole step  
- E to F: Half step
- F to G: Whole step
- G to A: Whole step
- A to B: Whole step
- B to C: Half step

Other common scales include:
- **Natural Minor**: W-H-W-W-H-W-W
- **Harmonic Minor**: W-H-W-W-H-W+H-H
- **Pentatonic**: Uses only 5 notes (e.g., C-D-E-G-A)"""
        
        return "I can help you with major scales! Try asking about a specific key like 'What is the C major scale?' or 'G major scale notes'."
    
    def _handle_chord_query(self, query: str) -> str:
        """Handle queries about chords."""
        query_lower = query.lower()
        
        # Check for specific chord types
        for chord_type, pattern in self.chord_patterns.items():
            if chord_type in query_lower:
                interval_names = self._pattern_to_intervals(pattern)
                return f"A {chord_type} chord contains these intervals: {', '.join(interval_names)}\nSemitone pattern: {pattern}"
                
        # General chord information
        return """**Basic Chord Types:**

🎵 **Major Chord**: Root, Major 3rd, Perfect 5th (e.g., C-E-G)
   - Happy, bright sound
   
🎵 **Minor Chord**: Root, Minor 3rd, Perfect 5th (e.g., C-Eb-G)
   - Sad, darker sound
   
🎵 **Diminished Chord**: Root, Minor 3rd, Diminished 5th (e.g., C-Eb-Gb)
   - Tense, unstable sound
   
🎵 **Augmented Chord**: Root, Major 3rd, Augmented 5th (e.g., C-E-G#)
   - Mysterious, floating sound

**7th Chords** add the 7th interval for richer harmony:
- **Major 7th**: Major chord + Major 7th
- **Minor 7th**: Minor chord + Minor 7th  
- **Dominant 7th**: Major chord + Minor 7th"""
    
    def _handle_interval_query(self, query: str) -> str:
        """Handle queries about musical intervals."""
        return """**Musical Intervals** are the distance between two notes:

**Perfect Intervals:**
- Unison (0 semitones)
- Perfect 4th (5 semitones) - C to F
- Perfect 5th (7 semitones) - C to G
- Octave (12 semitones) - C to C

**Major Intervals:**
- Major 2nd (2 semitones) - C to D
- Major 3rd (4 semitones) - C to E
- Major 6th (9 semitones) - C to A
- Major 7th (11 semitones) - C to B

**Minor Intervals** (1 semitone smaller than major):
- Minor 2nd (1 semitone) - C to Db
- Minor 3rd (3 semitones) - C to Eb
- Minor 6th (8 semitones) - C to Ab
- Minor 7th (10 semitones) - C to Bb

**Augmented/Diminished:**
- Augmented 4th / Diminished 5th (6 semitones) - C to F# / C to Gb"""
    
    def _handle_key_signature_query(self, query: str) -> str:
        """Handle queries about key signatures."""
        return """**Key Signatures** indicate which notes are sharp or flat throughout a piece:

**Sharp Keys** (Order of sharps: F# C# G# D# A# E# B#):
- G major: 1 sharp (F#)
- D major: 2 sharps (F#, C#)
- A major: 3 sharps (F#, C#, G#)
- E major: 4 sharps (F#, C#, G#, D#)
- B major: 5 sharps (F#, C#, G#, D#, A#)

**Flat Keys** (Order of flats: Bb Eb Ab Db Gb Cb Fb):
- F major: 1 flat (Bb)
- Bb major: 2 flats (Bb, Eb)  
- Eb major: 3 flats (Bb, Eb, Ab)
- Ab major: 4 flats (Bb, Eb, Ab, Db)
- Db major: 5 flats (Bb, Eb, Ab, Db, Gb)

**Relative Minor Keys** share the same key signature but start on the 6th degree of the major scale (e.g., A minor has the same key signature as C major)."""
    
    def _explain_circle_of_fifths(self) -> str:
        """Explain the circle of fifths."""
        return f"""**Circle of Fifths** is a visual representation of key signatures and their relationships:

🔄 **Clockwise** (ascending 5ths): {' → '.join(self.circle_of_fifths)}

**Key Relationships:**
- Each step clockwise adds one sharp (or removes one flat)
- Each step counter-clockwise adds one flat (or removes one sharp)
- Keys next to each other share 6 out of 7 notes
- Opposite keys (tritone apart) are most distant harmonically

**Useful for:**
- Finding relative and parallel keys
- Understanding chord progressions
- Modulation (key changes)
- Harmonic analysis"""
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general music theory queries."""
        return f"""I understand you're asking about: "{query}"

🎵 **I can help with these music theory topics:**

**Scales**: Major, minor, pentatonic, modes
**Chords**: Triads, 7th chords, extensions, inversions  
**Intervals**: Perfect, major, minor, augmented, diminished
**Key Signatures**: Sharps, flats, circle of fifths
**Harmony**: Chord progressions, functional harmony
**Rhythm**: Time signatures, note values, syncopation

**Examples of questions I can answer:**
- "What is the G major scale?"
- "Explain minor chords"
- "What are musical intervals?"
- "How does the circle of fifths work?"

What specific music theory topic would you like to explore?"""
    
    def _pattern_to_intervals(self, pattern):
        """Convert semitone pattern to interval names."""
        interval_names = []
        for semitones in pattern:
            if semitones == 0:
                interval_names.append("Root")
            elif semitones == 1:
                interval_names.append("Minor 2nd")
            elif semitones == 2:
                interval_names.append("Major 2nd")
            elif semitones == 3:
                interval_names.append("Minor 3rd")
            elif semitones == 4:
                interval_names.append("Major 3rd")
            elif semitones == 5:
                interval_names.append("Perfect 4th")
            elif semitones == 6:
                interval_names.append("Tritone")
            elif semitones == 7:
                interval_names.append("Perfect 5th")
            elif semitones == 8:
                interval_names.append("Minor 6th")
            elif semitones == 9:
                interval_names.append("Major 6th")
            elif semitones == 10:
                interval_names.append("Minor 7th")
            elif semitones == 11:
                interval_names.append("Major 7th")
            else:
                interval_names.append(f"{semitones} semitones")
        return interval_names
    
    def run(self, query: str) -> str:
        """Alias for execute() method."""
        return self.execute(query)
