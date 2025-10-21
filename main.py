from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Story Generator API", version="1.0.0")

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

print("OpenAI API Key found:", api_key[:10] + "..." if api_key else "Not found")

class StoryElement(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None

class Chapter(BaseModel):
    title: str
    content: str
    word_count: int
    chapter_number: int

class StoryRequest(BaseModel):
    story_elements: List[StoryElement] = Field(..., min_items=1)
    word_limit: int = Field(..., ge=100, le=10000)

class StoryResponse(BaseModel):
    title: str
    chapters: List[Chapter]
    total_word_count: int
    generated_elements: Dict[str, Any]

async def call_openai_api(messages: List[Dict], max_tokens: int = 8000) -> str:
    """Direct API call to OpenAI using httpx"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=500, detail=f"OpenAI API HTTP error: {e.response.status_code}")
        except Exception as e:
            print(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API request failed: {str(e)}")

async def generate_story_chunk(prompt: str, max_tokens: int = 4000) -> str:
    """Generate a chunk of the story using direct OpenAI API call"""
    try:
        print(f"Sending request to OpenAI with {len(prompt)} characters...")
        
        messages = [
            {"role": "system", "content": "You are a creative fiction writer. Write engaging, detailed stories based on the provided questions and answers."},
            {"role": "user", "content": prompt}
        ]
        
        return await call_openai_api(messages, max_tokens)
    
    except Exception as e:
        print(f"OpenAI API error details: {str(e)}")
        raise

async def generate_chunk_with_retry(prompt: str, target_words: int, max_retries: int = 3) -> str:
    """Generate a story chunk with retry logic to meet word count"""
    for attempt in range(max_retries):
        try:
            # Adjust tokens based on target words (roughly 1.3 tokens per word)
            estimated_tokens = min(int(target_words * 1.3) + 200, 4000)
            
            chunk = await generate_story_chunk(prompt, estimated_tokens)
            word_count = len(chunk.split())
            
            print(f"Chunk generated: {word_count} words (target: {target_words})")
            
            # If we're within 20% of target, accept it
            if word_count >= target_words * 0.8:
                return chunk
            else:
                print(f"Chunk too short ({word_count} words), retrying... (attempt {attempt + 1})")
                # Modify prompt to ask for more content
                retry_prompt = prompt + f"\n\nIMPORTANT: The previous response was too short. Please write at least {target_words} words for this section. Add more detail, description, and depth to reach the required length."
                prompt = retry_prompt
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    # If all retries fail, return whatever we have
    return chunk

async def extract_chapter_titles(outline: str) -> List[str]:
    """Extract chapter titles from the outline"""
    try:
        title_prompt = f"""
        Extract the chapter titles from this story outline. Return ONLY a JSON array of chapter titles, nothing else.
        
        STORY OUTLINE:
        {outline}
        
        Example output: ["Chapter 1: Childhood Beginnings", "Chapter 2: School Days", ...]
        
        Return only the JSON array:
        """
        
        titles_json = await generate_story_chunk(title_prompt, 500)
        
        # Clean the response to extract JSON
        titles_json = titles_json.strip()
        if titles_json.startswith('```json'):
            titles_json = titles_json[7:]
        if titles_json.endswith('```'):
            titles_json = titles_json[:-3]
        
        chapter_titles = json.loads(titles_json)
        return chapter_titles
        
    except Exception as e:
        print(f"Error extracting chapter titles: {e}")
        # Fallback: generate default chapter titles
        return [
            "Chapter 1: Early Childhood Memories",
            "Chapter 2: Family and Foundations", 
            "Chapter 3: School Days and Friendships",
            "Chapter 4: Dreams and Aspirations",
            "Chapter 5: Love and Relationships",
            "Chapter 6: Life Lessons and Growth",
            "Chapter 7: Reflections and Future"
        ]

async def generate_long_story(elements: List[StoryElement], word_limit: int) -> Dict[str, Any]:
    """Generate a long story with separate chapters"""
    
    try:
        # Prepare Q&A summary
        qa_pairs = []
        for i, element in enumerate(elements, 1):
            if element.question and element.answer:
                qa_pairs.append(f"{i}. Q: {element.question}\nA: {element.answer}")
        
        if not qa_pairs:
            raise HTTPException(status_code=400, detail="No valid question-answer pairs provided")
        
        qa_text = "\n\n".join(qa_pairs)
        
        print(f"Processing {len(qa_pairs)} Q&A pairs for {word_limit} word story...")
        
        # Generate overall outline first
        outline_prompt = f"""
        Create a detailed chapter outline for a {word_limit} word life story based on these questions and answers:
        
        QUESTIONS AND ANSWERS:
        {qa_text}
        
        Create a 5-7 chapter outline with clear chapter titles that covers:
        1. Childhood and early memories
        2. Family relationships and background  
        3. Education and personal development
        4. Friendships and social life
        5. Love and romantic relationships
        6. Career and personal achievements
        7. Life lessons and future outlook
        
        Make each chapter title descriptive and engaging.
        Return the outline with clear and brief descriptions. Make sure it's engaging and attracts the reader.
        """
        
        print("Generating story outline...")
        outline = await generate_story_chunk(outline_prompt, 2000)
        print("Outline generated successfully")
        
        # Extract chapter titles from outline
        chapter_titles = await extract_chapter_titles(outline)
        num_chapters = len(chapter_titles)
        
        print(f"Extracted {num_chapters} chapter titles: {chapter_titles}")
        
        # Calculate words per chapter
        base_words_per_chapter = word_limit // num_chapters
        remaining_words = word_limit
        
        chapters = []
        current_context = ""
        
        print(f"Starting chapter generation for {num_chapters} chapters...")
        
        for chapter_num, chapter_title in enumerate(chapter_titles, 1):
            # Calculate target words for this chapter
            chapters_remaining = num_chapters - chapter_num + 1
            target_words = min(base_words_per_chapter, remaining_words // chapters_remaining)
            
            chapter_prompt = f"""
            STORY OUTLINE:
            {outline}
            
            KEY ELEMENTS TO INCORPORATE:
            {qa_text}
            
            PREVIOUS STORY CONTEXT:
            {current_context if current_context else "This is the beginning of the story."}
            
            
            INSTRUCTIONS:
            - Write approximately {target_words} words for this chapter
            - Do not include chapter title
            - Continue the story naturally from the previous context
            - Focus on the themes appropriate for this chapter
            - Incorporate relevant Q&A elements organically
            - Write in a descriptive, emotional style
            - Ensure this chapter feels complete but leads naturally to the next
            """
            
            if chapter_num == 1:
                chapter_prompt += """
                Start the story from childhood. Establish the main character, setting, and early memories.
                Make this chapter engaging and set the tone for the entire story.
                """
            elif chapter_num == num_chapters:
                chapter_prompt += """
                This is the final chapter. Provide meaningful closure, reflect on life lessons,
                and show how all experiences shaped the person. End with hope and forward-looking thoughts.
                Make this chapter a satisfying conclusion to the entire story.
                """
            else:
                chapter_prompt += f"""
                This is a middle chapter. Continue developing the narrative with depth and detail.
                Build on previous chapters and set up developments for future chapters.
                """
            
            print(f"Generating chapter {chapter_num}/{num_chapters}: '{chapter_title}' (target: {target_words} words)...")
            
            # Generate chapter content with retry logic
            chapter_content = await generate_chunk_with_retry(chapter_prompt, target_words)
            chapter_word_count = len(chapter_content.split())
            
            # Create chapter object
            chapter = Chapter(
                title=chapter_title,
                content=chapter_content,
                word_count=chapter_word_count,
                chapter_number=chapter_num
            )
            chapters.append(chapter)
            
            # Update context for next chapter (use last chapter for continuity)
            current_context = chapter_content
            current_total_words = sum(chap.word_count for chap in chapters)
            remaining_words = word_limit - current_total_words
            
            print(f"Chapter {chapter_num} completed: {chapter_word_count} words. Total so far: {current_total_words} words")
            
            # Add delay between chapters
            if chapter_num < num_chapters:
                await asyncio.sleep(2)
        
        # Check if we need additional content to reach word limit
        current_total_words = sum(chapter.word_count for chapter in chapters)
        
        if current_total_words < word_limit * 0.9:
            print(f"Story is short ({current_total_words} words), generating additional content...")
            
            additional_prompt = f"""
            STORY OUTLINE:
            {outline}
            
            COMPLETE STORY SO FAR:
            {' '.join([chap.content for chap in chapters])[-3000:]}
            
            ADDITIONAL CONTENT NEEDED:
            We need approximately {word_limit - current_total_words} more words to complete the story.
            
            Please write an additional chapter or epilogue that:
            - Expands on existing themes from the story
            - Adds depth to character development
            - Provides additional reflections or memories
            - Enhances the emotional journey
            - Maintains consistency with the existing story
            
            Write this additional content that seamlessly continues from the current story.
            """
            
            additional_content = await generate_chunk_with_retry(additional_prompt, word_limit - current_total_words)
            additional_word_count = len(additional_content.split())
            
            # Create additional chapter
            additional_chapter = Chapter(
                title="Epilogue: Final Reflections",
                content=additional_content,
                word_count=additional_word_count,
                chapter_number=len(chapters) + 1
            )
            chapters.append(additional_chapter)
            
            current_total_words += additional_word_count
            print(f"Additional content added: {additional_word_count} words. New total: {current_total_words} words")
        
        # Generate overall story title
        title_prompt = f"""
        Based on this life story with the following chapters, create an engaging, heartfelt title:
        
        CHAPTERS:
        {[chap.title for chap in chapters]}
        
        KEY THEMES: Childhood memories, family, friendships, love, personal growth, life lessons
        
        Return only the title without quotes. Make it emotional and memorable.
        """
        story_title = await generate_story_chunk(title_prompt, 100)
        
        final_total_words = sum(chapter.word_count for chapter in chapters)
        
        print(f"Story generation completed: {final_total_words} words across {len(chapters)} chapters (target: {word_limit})")
        
        return {
            "title": story_title.strip('"\''),
            "chapters": chapters,
            "total_word_count": final_total_words,
            "generated_elements": {
                "qa_pairs_used": len(qa_pairs),
                "total_chapters": len(chapters),
                "chapter_titles": [chap.title for chap in chapters],
                "outline": outline,
                "target_achieved": final_total_words >= word_limit * 0.9
            }
        }
        
    except Exception as e:
        print(f"Error in generate_long_story: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story_endpoint(request: StoryRequest):
    """
    Generate a story based on questions and answers input
    """
    try:
        print(f"Received request for {len(request.story_elements)} Q&A pairs, {request.word_limit} words")
        result = await generate_long_story(
            elements=request.story_elements,
            word_limit=request.word_limit
        )
        
        return result
        
    except Exception as e:
        print(f"Error in generate_story endpoint: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Story Generator API", 
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-story": "Generate stories from Q&A input"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Test endpoint to check OpenAI connection
@app.get("/test-openai")
async def test_openai():
    """Test OpenAI connection with a simple prompt"""
    try:
        test_prompt = "Write a one sentence story about a cat."
        messages = [
            {"role": "user", "content": test_prompt}
        ]
        
        response = await call_openai_api(messages, 50)
        
        return {
            "status": "success",
            "response": response
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")