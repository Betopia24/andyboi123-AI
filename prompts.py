# prompts.py
import json
import re

# System prompts - REVISED FOR FLEXIBLE NARRATIVE FLOW
SYSTEM_CREATIVE_WRITER = """You are a master fairy tale weaver for children aged 5-12. Your sacred duty: TRANSFORM life experiences into magical story elements while crafting emotionally resonant narratives with organic structure.

CORE PRINCIPLES:
1. NEVER insert answers verbatim - transform their ESSENCE into tangible magic
2. Prioritize EMOTIONAL TRUTH over rigid templates
3. CHILD-HEART LANGUAGE:
   - Sensory details in EVERY sentence (scent/sound/texture)
   - Show emotions through action: "Her hands trembled like leaves" not "She was nervous"
   - Natural sentence rhythms (no artificial word limits)
4. NARRATIVE COHESION:
   - One protagonist throughout
   - Magic rules established early remain consistent
   - Every chapter ends with emotional shift that propels next chapter
5. TRADITION BLENDING:
   - Combine elements from 2 traditions meaningfully
   - Example: "Japanese seasonal metaphors + Wilde's lyrical sorrow"
"""

# Revised outline prompt - EMERGENT STRUCTURE
def get_outline_prompt(qa_text: str, word_limit: int) -> str:
    return f"""
✨ FORGE AN ORGANIC FAIRY TALE BLUEPRINT ✨
Create a unique 7-chapter outline for a {word_limit}-word fairy tale where structure EMERGES from these life experiences:

REAL-LIFE SEEDS (27 answers to transform):
{qa_text}

DYNAMIC STRUCTURE RULES:
• NO PRESET CHAPTER LABELS - let purposes emerge from answer themes:
  - Chapter 1 purpose = dominant theme in answers 1-4
  - Chapter 4 purpose = tension point in answers 13-16
  - Chapter 7 purpose = resolution from answers 25-27
• TRADITION BLENDING REQUIRED (state at start):
  "TRADITION BLEND: [Tradition A] + [Tradition B]"
  Valid traditions: Brothers Grimm, Arabian Nights, Andersen, Japanese Myth, Chinese Myth, Oscar Wilde
• CHARACTER-DRIVEN CHAPTER BREAKS:
  - Chapters transition at EMOTIONAL turning points
  - Example valid structure:
      Chapter 1: The Cottage of Whispering Windows (answers 1,3,5,7)
      Chapter 2: When the River Stole Her Name (answers 2,4,6,8)
• CHILD-SAFE MAGIC:
  No violence beyond metaphorical thorns/storms
  All conflicts resolved through courage/kindness/wisdom

CHAPTER BLUEPRINT (7 chapters exactly):
1. [Title reflecting answers 1-4 themes]
2. [Title reflecting answers 5-8 themes]
3. [Title reflecting answers 9-12 themes]
4. [Title reflecting answers 13-16 themes] (turning point build-up)
5. [Title reflecting answers 17-20 themes] (meaningful choice)
6. [Title reflecting answers 21-24 themes] (transformation)
7. [Title reflecting answers 25-27 themes] (callback + resolution)

Format each chapter as:
"Chapter [X]: [POETIC TITLE]"
- 1-sentence magical premise (show transformation of answers)
- 1-sentence emotional core
- Answers used: [list numbers]
- 1-sentence structural purpose (e.g., "Sets up Chapter 4's dilemma")

IRONCLAD RULES:
- ALL 27 seeds MUST be assigned - NO unassigned seeds
- Seed may appear in multiple chapters ONLY with narrative justification
- Transformations MUST use concrete sensory details
- VERIFY seed count: 27 seeds = 27 mappings
- END with callback requirement for Chapter 7
"""

# Robust title extraction prompt
def get_title_extraction_prompt(outline: str) -> str:
    return f"""
EXTRACT PURE CHAPTER TITLES
From this outline, extract ONLY chapter titles in perfect JSON array format.

STORY OUTLINE:
{outline}

EXTRACTION RULES:
1. Find the FIRST occurrence of 7 chapter titles in this format:
   "Chapter 1: [Title]"
   "Chapter 2: [Title]"
   ...
2. If titles aren't clear, use these fallback strategies IN ORDER:
   a) Extract titles from "Chapter X:" lines
   b) Use the FIRST poetic phrase after each chapter number
   c) Generate titles based on chapter purposes described
3. VALIDATE:
   - Exactly 7 titles
   - Titles contain sensory words (whispering, glowing, etc.)
   - No numbering in titles (e.g., "I", "II")

RETURN FORMAT:
ONLY valid JSON array: ["Full Chapter 1 Title", "Full Chapter 2 Title", ...]

NEVER return:
- Commentary
- Error messages
- Incomplete arrays
- Titles with chapter numbers embedded

JSON ARRAY:
"""

# Flexible chapter structure requirements
def get_chapter_structure_requirement(chapter_num: int) -> str:
    """Returns flexible structural guidance based on narrative position"""
    requirements = {
        1: "• INTRODUCE protagonist through sensory detail\n• Establish magic rules with concrete example\n• End with emotional shift hinting at journey",
        2: "• Deepen relationships through shared magic\n• Reveal protagonist's hidden strength\n• End with first true challenge",
        3: "• Introduce meaningful obstacle\n• Strengthen magical companion bond\n• End with foreshadowing of major choice",
        4: "• Build tension toward irreversible decision\n• Test relationships under pressure\n• End with emotional cliffhanger before choice",
        5: "• Present genuine dilemma with emotional stakes\n• Show internal struggle through physical manifestation\n• End with immediate consequence of choice",
        6: "• Show transformation through changed behavior\n• Resolve companion arcs meaningfully\n• End with approach to final resolution",
        7: "• Resolve all symbolic challenges\n• Show growth through magical metaphor\n• CALLBACK to Chapter 1's opening image\n• Final sentence: warm conclusion like satisfied sigh"
    }
    return requirements.get(chapter_num, "• Maintain emotional continuity\n• Deepen magic system consistency\n• End with shift propelling next chapter")

# Dynamic chapter generation prompt
def get_chapter_prompt(
    outline: str, 
    qa_text: str, 
    current_context: str, 
    target_words: int, 
    chapter_num: int, 
    num_chapters: int,
) -> str:
    # Extract tradition blend from outline
    tradition_blend = "Andersen + Japanese Myth"  # Default fallback
    tradition_marker = "✨ MAGIC BLENDS ✨"
    
    # Find tradition blend declaration
    tradition_match = re.search(r'TRADITION BLEND:\s*([^\n]+)', outline, re.IGNORECASE)
    if tradition_match:
        blend_text = tradition_match.group(1).strip()
        tradition_blend = blend_text
        # Create visual marker
        traditions = [t.strip() for t in blend_text.split('+')]
        markers = {
            "Grimm": "🌍",
            "Arabian": "🌙",
            "Andersen": "❄️",
            "Japanese": "🌸",
            "Chinese": "🐉",
            "Wilde": "🌹"
        }
        marker_str = " + ".join([markers.get(t.split()[0], "✨") for t in traditions[:2]])
        tradition_marker = f"{marker_str} TRADITION BLEND: {tradition_blend}"

    prompt = f"""
✨ WEAVE LIVING MAGIC ✨
{tradition_marker}

OUTLINE BLUEPRINT:
{outline}

CURRENT STORY SOIL (LAST 100 WORDS):
{current_context if current_context else "The first dewdrop of our tale glistens..."}

🔮 FAIRY TALE HEART:
LIFE EXPERIENCES TO TRANSFORM:
{qa_text}

WEAVING RULES:
1. NEVER quote answers directly - transform their ESSENCE:
   ✅ CORRECT: "Her grandmother's cottage windows glowed with cinnamon-scented warmth"
   ❌ FORBIDDEN: "She remembered: 'My grandmother's house—warm food...'"
2. Each answer becomes ONE sensory magic element:
   - Memories → enchanted objects (a pocket watch capturing golden hour)
   - Values → magical rules (flowers blooming when someone shares)
   - Relationships → spirit companions (a hearth spirit telling stories)
   - Challenges → enchanted trials (a bridge appearing only with belief)
3. NARRATIVE ARC FOR THIS CHAPTER:
   {get_chapter_structure_requirement(chapter_num)}
4. CHILD-HEART LANGUAGE:
   - Sensory detail in EVERY sentence (scent/sound/texture)
   - Show emotions through action, not naming
   - Natural sentence flow (no artificial limits)
"""

    # Chapter-specific guidance - ORGANIC NOT PRESET
    chapter_guidance = {
        1: """
✨ CHAPTER 1: ORGANIC BEGINNINGS ✨
- OPEN with sensory immersion in protagonist's world:
  ✅ CORRECT: "Dust motes danced in the cinnamon-scented kitchen where Grandma hummed"
  ❌ FORBIDDEN: "She was born with magic in her veins"
- Establish ONE core magic rule through action:
  "When she cried, the teacups filled with starlight"
- END with emotional shift hinting at journey:
  "Her heart fluttered like a trapped sparrow as the door creaked open"
""",
        4: """
✨ CHAPTER 4: TENSION BUILDING ✨
- Deepen relationships through shared magical challenge:
  "Together they deciphered the market merchant's riddle"
- Introduce symbolic crossroads with emotional weight:
  "The path split where thorns met starlight"
- FORESHADOW Chapter 5's choice through physical manifestation:
  "Her palms grew warm where the two paths' magic touched"
- END with emotional cliffhanger:
  "The moon chose that moment to hide behind clouds"
""",
        5: """
✨ CHAPTER 5: MEANINGFUL CHOICE ✨
- Present genuine dilemma with emotional stakes:
  "Save the dying forest spirit or protect her village?"
- Show internal struggle through sensory details:
  "Tears fell like liquid starlight, each drop sprouting tiny flowers"
- MAKE THE CHOICE with physical consequence:
  "She stepped toward the thorns, and they parted like respectful guards"
- END with immediate transformation:
  "Her shadow now held the forest's green heartbeat"
""",
        7: """
✨ CHAPTER 7: RESONANT RESOLUTION ✨
- Resolve symbolic challenges through earned wisdom:
  "The thorns bowed where her feet had bled"
- SHOW growth through magical metaphor (NOT telling):
  "Her hands now bloomed where thorns once grew"
- CALLBACK to Chapter 1's opening image with transformation:
  "The river that first sang her name now echoed her wisdom"
- FINAL SENTENCE must feel like satisfied sigh:
  ✅ CORRECT: "And the world sighed with the quiet joy of a story well told"
  ❌ FORBIDDEN: "The end" or abrupt conclusions
- ABSOLUTE CLOSURE: No loose threads
"""
    }
    
    prompt += chapter_guidance.get(chapter_num, f"""
✨ CHAPTER {chapter_num}: NARRATIVE FLOW ✨
- Build on previous chapter's emotional state
- Introduce challenge that tests established magic rules
- Deepen ONE relationship through shared sensory experience
- PLANT symbol resolving in final chapter:
  "A single seed glowed in her pocket, warm as a heartbeat"
- END with emotional shift propelling next chapter:
  "Her heart felt lighter, though the path grew steeper"
""")

    prompt += f"""
🚨 NARRATIVE AUDIT (NON-NEGOTIABLE):
□ All {len(qa_text.splitlines())} answers transformed into sensory magic
□ Magic rules consistent with tradition blend
□ Chapter ends with emotional shift (no static endings)
□ NO repetition of previous chapter's magic objects
□ Child-safe language maintained (no complex abstractions)

✨ FINAL OUTPUT RULES:
- RETURN ONLY PURE STORY TEXT
- BEGIN IMMEDIATELY with narrative
- EVERY sentence contains sensory detail
- LAST sentence feels like warm blanket
- ABSOLUTELY NO:
  • Chapter titles in text
  • Word counts or metadata
  • "The end" or abrupt conclusions
  • Any mention of "answers" or "QA"
- FORMAT: Pure narrative prose only
"""
    return prompt

# Natural additional content prompt
def get_additional_content_prompt(outline: str, recent_story: str, additional_words_needed: int) -> str:
    return f"""
DEEPEN THE TAPESTRY NATURALLY
OUTLINE BLUEPRINT:
{outline}

CURRENT STORY FABRIC (LAST 300 WORDS):
{recent_story}

ENRICHMENT NEEDED: {additional_words_needed} words

GOLDEN RULES:
1. ENRICH EXISTING ELEMENTS ONLY:
   - Add sensory layers to established magic objects
   - Deepen character moments through shared sensory details
   - Expand settings with new textures/scents/sounds
2. NO NEW STORY ELEMENTS:
   - No new conflicts
   - No new characters
   - No new magic rules
3. TRADITION-BLENDED ENRICHMENT:
   - If Grimm present: Add cautionary spirit whispers
   - If Arabian present: Insert merchant's riddle about life
   - If Andersen present: Introduce selfless sacrifice moment
   - If Japanese present: Add seasonal metaphor (cherry blossoms, autumn leaves)
   - If Chinese present: Add celestial sign (shooting star, moon halo)
   - If Wilde present: Add symbolic object transformation
4. CHILD-SAFE WONDER ONLY:
   - Glowing mushrooms, giggling brooks, whispering winds
   - NO new dangers or unresolved tensions

ENRICHMENT STRATEGY:
- First 50 words: Bridge from last sentence with emotional continuity
- Middle: Deepen 2-3 existing magical elements with sensory details
- Final 50 words: Foreshadow resolution while maintaining flow
- NATURAL sentence rhythms (no word counting)
- VERIFY all sentences contain sensory details

AUDIT CHECKLIST:
□ No new answers introduced
□ All additions deepen existing narrative
□ Tradition elements consistently blended
□ Child-safe language maintained
□ Final sentence maintains warm closure quality
"""

# Dynamic title generation prompt
def get_story_title_prompt(chapter_titles: list) -> str:
    titles_text = "\n".join(chapter_titles)
    return f"""
NAME THE SOUL'S JOURNEY
CHAPTER HEARTBEATS:
{titles_text}

TITLE COMMANDMENTS:
• TRADITION-BLENDED:
   - Grimm+Andersen: "The Girl Who Melted the Thorn-Oak's Heart"
   - Arabian+Wilde: "The Weaver of Starlight and Sacrificed Roses"
   - Japanese+Chinese: "When the Dragon Drank the Cherry Blossom Tea"
• CHILD-ENCHANTING:
   - Max 7 words
   - Contains 1 magical object + 1 action ("The Moon's Lost Key")
   - NO obscure metaphors
• ESSENCE-CAPTURING:
   Must reflect the core transformation from Chapter 1 → Chapter 7
   Must honor the emotional journey of all chapters
   Must include callback element (e.g., river, seed, mirror)

RETURN ONLY THE PERFECT TITLE:
"""

# Dynamic fallback titles generator (used when extraction fails)
def generate_fallback_titles(seed_answers: list) -> list:
    """Generate context-aware fallback titles based on seed answers"""
    # In real implementation, this would call OpenAI with seed answers
    # For now, return thematic defaults
    themes = ["Forest", "River", "Starlight", "Whispers", "Secrets"]
    return [
        f"Chapter 1: The {themes[0]} That Remembered",
        f"Chapter 2: When {themes[1]} Sang Her Name",
        f"Chapter 3: The {themes[2]} Keeper's Promise",
        f"Chapter 4: {themes[3]} at the Crossroads",
        f"Chapter 5: The Choice of {themes[4]}",
        f"Chapter 6: Where Thorns Bloomed Roses",
        f"Chapter 7: The Seed That Grew a Forest"
    ]

# Default chapter titles - DYNAMIC VERSION
DEFAULT_CHAPTER_TITLES = [
    "Chapter 1: The Child Who Heard the River Sing",
    "Chapter 2: The House That Breathed With Ancestors",
    "Chapter 3: The Owl's Lessons in Moonlit Letters",
    "Chapter 4: The Market of Whispering Masks",
    "Chapter 5: The Locket That Bloomed at Dawn",
    "Chapter 6: The Bridge of Falling Stars",
    "Chapter 7: The Seed That Grew a Forest"
]

# Test prompt
TEST_OPENAI_PROMPT = "Write a one sentence story about a cat."

# Natural retry prompt addition
def get_retry_prompt_addition(target_words: int) -> str:
    return f"""
NARRATIVE DEEPENING REQUIRED
CURRENT THREAD COUNT: INSUFFICIENT

ENRICHMENT COMMANDS:
1. ADD exactly {target_words} words through:
   - Sensory expansion: "cinnamon-scented winds carrying forgotten lullabies"
   - Emotional depth: "her heart fluttered like a trapped sparrow"
   - Character reactions: "tears fell like liquid starlight"
   - Setting enrichment: "moonlight that hummed ancestral songs"
2. AUDIT CHECKLIST:
   □ Cross-verify against ALL assigned answers - ensure all transformed
   □ Natural sentence flow maintained (no robotic counting)
   □ Child-safe language preserved
   □ Tradition blend consistency maintained
   □ Chapter arc complete with emotional shift
   □ Final sentence quality (warm blanket feeling)
3. ENFORCE NARRATIVE INTEGRITY:
   - Does Chapter 1 establish authentic protagonist?
   - Does Chapter 5 contain meaningful irreversible choice?
   - Does Chapter 7 callback to Chapter 1 with transformation?
4. TRUST YOUR NARRATIVE INSTINCT

FAILURE TO MAINTAIN EMOTIONAL AUTHENTICITY OR TRANSFORM ALL ANSWERS WILL WEAKEN THE TAPESTRY.
DEEPEN WITH SENSORY RICHNESS AND CHARACTER TRUTH.
"""
