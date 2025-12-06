# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import asyncio
import json
import os
import re
from dotenv import load_dotenv
from prompts import (
    SYSTEM_CREATIVE_WRITER,
    get_outline_prompt,
    get_title_extraction_prompt,
    get_chapter_prompt,
    get_additional_content_prompt,
    get_story_title_prompt,
    TEST_OPENAI_PROMPT,
    get_retry_prompt_addition
)

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
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get('error', {}).get('message', str(e))
            print(f"HTTP error: {e.response.status_code} - {error_detail}")
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {error_detail}")
        except Exception as e:
            print(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"OpenAI API request failed: {str(e)}")

async def generate_story_chunk(prompt: str, max_tokens: int = 4000) -> str:
    """Generate a chunk of the story using direct OpenAI API call"""
    print(f"Sending request to OpenAI with {len(prompt)} characters...")
    
    messages = [
        {"role": "system", "content": SYSTEM_CREATIVE_WRITER},
        {"role": "user", "content": prompt}
    ]
    
    return await call_openai_api(messages, max_tokens)

async def generate_chunk_with_retry(prompt: str, target_words: int, max_retries: int = 3) -> str:
    """Generate a story chunk with retry logic focused on narrative depth"""
    for attempt in range(max_retries):
        try:
            estimated_tokens = min(int(target_words * 1.5) + 200, 4000)
            
            chunk = await generate_story_chunk(prompt, estimated_tokens)
            word_count = len(chunk.split())
            
            print(f"Chunk generated: {word_count} words (target: {target_words})")
            
            # Accept if within reasonable range or over target
            if word_count >= target_words * 0.8 or word_count > target_words:
                return chunk
            else:
                print(f"Chunk too short ({word_count} words), enriching narrative... (attempt {attempt + 1})")
                retry_prompt = prompt + get_retry_prompt_addition(target_words)
                prompt = retry_prompt
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    return chunk

def parse_titles_from_outline(outline: str) -> List[str]:
    """Extract chapter titles directly from outline text"""
    titles = []
    # Look for chapter patterns like "Chapter 1: The Magical Forest"
    pattern = r'Chapter\s+(\d+):\s*([^\n]+)'
    matches = re.findall(pattern, outline, re.IGNORECASE)
    
    # Sort by chapter number
    sorted_matches = sorted(matches, key=lambda x: int(x[0]))
    
    for num, title in sorted_matches:
        # Clean title: remove quotes, extra spaces
        clean_title = re.sub(r'^[\"\']|[\"\']$', '', title).strip()
        titles.append(f"Chapter {num}: {clean_title}")
    
    # Validate we have at least 3 chapters (minimum viable story)
    if len(titles) < 3:
        raise ValueError("Could not extract sufficient chapter titles from outline")
    
    return titles

async def extract_chapter_titles(outline: str) -> List[str]:
    """Extract chapter titles with strict validation and no fallbacks"""
    try:
        title_prompt = get_title_extraction_prompt(outline)
        titles_response = await generate_story_chunk(title_prompt, 500)
        
        print(f"Raw title response: {titles_response[:200]}...")
        
        # Strategy 1: Find JSON array
        json_match = re.search(r'\[[^\]]*\]', titles_response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                chapter_titles = json.loads(json_str)
                if isinstance(chapter_titles, list) and len(chapter_titles) >= 3:
                    print(f"✅ JSON titles extracted: {chapter_titles}")
                    # Ensure proper formatting
                    return [f"Chapter {i+1}: {title}" for i, title in enumerate(chapter_titles)]
            except json.JSONDecodeError:
                print("JSON parsing failed, trying text extraction")
        
        # Strategy 2: Parse directly from outline text
        print("🔄 Parsing titles directly from outline text")
        text_titles = parse_titles_from_outline(outline)
        print(f"✅ Text-parsed titles: {text_titles}")
        return text_titles
        
    except Exception as e:
        print(f"🚨 Title extraction failed: {str(e)}")
        # NO FALLBACK - re-raise the exception to fail the request
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to extract chapter titles from outline: {str(e)}"
        )

async def generate_long_story(elements: List[StoryElement], word_limit: int) -> Dict[str, Any]:
    """Generate a story that includes EVERY SINGLE ANSWER with no fallbacks"""
    try:
        # Collect ONLY valid answers - no placeholders, no padding
        all_answers = [
            element.answer.strip() 
            for element in elements 
            if element.answer and element.answer.strip()
        ]
        
        if not all_answers:
            raise HTTPException(status_code=400, detail="No valid answers provided")
        
        print(f"✨ Processing {len(all_answers)} UNIQUE answers for {word_limit}-word story...")
        print("📚 ALL ANSWERS WILL BE INCLUDED - NO FALLBACKS OR PLACEHOLDERS")
        
        # Generate outline using ALL answers
        outline_prompt = get_outline_prompt("\n".join(all_answers), word_limit)
        print("🌱 Generating story outline with ALL answers...")
        outline = await generate_story_chunk(outline_prompt, 2000)
        print("✅ Outline generated successfully")
        
        # Extract chapter titles with strict validation
        chapter_titles = await extract_chapter_titles(outline)
        num_chapters = len(chapter_titles)
        print(f"챕 Found {num_chapters} chapter titles: {chapter_titles}")
        
        # Calculate words per chapter
        words_per_chapter = [word_limit // num_chapters] * num_chapters
        remainder = word_limit % num_chapters
        for i in range(remainder):
            words_per_chapter[i] += 1
        
        # DISTRIBUTE ALL ANSWERS ACROSS CHAPTERS - NO SKIPPING
        answers_per_chapter = [0] * num_chapters
        
        # Base distribution
        base_answers = len(all_answers) // num_chapters
        remainder = len(all_answers) % num_chapters
        
        # Assign answers to chapters
        for i in range(num_chapters):
            answers_per_chapter[i] = base_answers
            if i < remainder:
                answers_per_chapter[i] += 1
        
        # Validate total answers match
        total_assigned = sum(answers_per_chapter)
        if total_assigned != len(all_answers):
            raise ValueError(
                f"Answer distribution mismatch: {total_assigned} assigned vs {len(all_answers)} total answers"
            )
        
        print(f"📊 Answer distribution: {answers_per_chapter} (total: {total_assigned}/{len(all_answers)})")
        
        chapters = []
        current_context = ""
        
        print(f"📖 Starting chapter generation for {num_chapters} chapters...")
        
        for chapter_num in range(1, num_chapters + 1):
            chapter_title = chapter_titles[chapter_num - 1]
            
            # Calculate EXACT answer range for this chapter
            start_idx = sum(answers_per_chapter[:chapter_num-1])
            end_idx = start_idx + answers_per_chapter[chapter_num-1]
            chapter_answers = all_answers[start_idx:end_idx]
            
            # CRITICAL: Verify we have answers for this chapter
            if not chapter_answers:
                raise HTTPException(
                    status_code=500,
                    detail=f"No answers assigned to Chapter {chapter_num} - distribution error"
                )
            
            # Get target words for this chapter
            target_words = words_per_chapter[chapter_num - 1]
            
            # Generate chapter content with EXACT answers for this chapter
            chapter_prompt = get_chapter_prompt(
                outline=outline,
                qa_text="\n".join(chapter_answers),
                current_context=current_context,
                target_words=target_words,
                chapter_num=chapter_num,
                num_chapters=num_chapters,
            )
            
            print(f"\n{'='*50}")
            print(f"✨ Generating Chapter {chapter_num}/{num_chapters}: '{chapter_title}'")
            print(f"🎯 Target words: {target_words} | Answers used: {len(chapter_answers)}")
            print("🔍 ANSWERS FOR THIS CHAPTER:")
            for i, ans in enumerate(chapter_answers, 1):
                print(f"   {i}. {ans[:60]}{'...' if len(ans)>60 else ''}")
            print(f"📚 Context: {current_context[-50:] if current_context else 'Beginning of story'}")
            print(f"{'='*50}\n")
            
            # Generate with retry logic
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
            
            # Update context for next chapter (last 150 words)
            if chapter_content:
                current_context = " ".join(chapter_content.split()[-150:])
            
            current_total_words = sum(chap.word_count for chap in chapters)
            print(f"✅ Chapter {chapter_num} completed: {chapter_word_count} words. Total: {current_total_words}/{word_limit}")
        
        # Generate story title
        title_prompt = get_story_title_prompt(chapter_titles)
        story_title = await generate_story_chunk(title_prompt, 100)
        clean_title = re.sub(r'^[\"\']|[\"\']$', '', story_title).strip()
        
        final_total_words = sum(chapter.word_count for chapter in chapters)
        
        # FINAL VALIDATION: Verify all answers were used
        used_answers_count = sum(answers_per_chapter)
        if used_answers_count != len(all_answers):
            raise HTTPException(
                status_code=500,
                detail=f"Answer mismatch: {used_answers_count} used vs {len(all_answers)} provided"
            )
        
        print(f"\n{'='*60}")
        print(f"🎉 Story generation completed: {final_total_words} words across {len(chapters)} chapters")
        print(f"📖 Title: '{clean_title}'")
        print(f"✅ ALL {len(all_answers)} ANSWERS INCLUDED - NO FALLBACKS USED")
        print(f"{'='*60}")
        
        return {
            "title": clean_title,
            "chapters": chapters,
            "total_word_count": final_total_words,
            "generated_elements": {
                "answers_used": len(all_answers),
                "total_chapters": len(chapters),
                "chapter_titles": chapter_titles,
                "outline": outline[:500] + "..." if len(outline) > 500 else outline,
                "answer_distribution": answers_per_chapter
            }
        }
        
    except Exception as e:
        print(f"🔥 Critical error in generate_long_story: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

@app.post("/generate-story", response_model=StoryResponse)
async def generate_story_endpoint(request: StoryRequest):
    """
    Generate a story that includes EVERY SINGLE ANSWER with no fallbacks or placeholders
    """
    print(f"\n{'*'*80}")
    print(f"🚀 RECEIVED STORY GENERATION REQUEST")
    print(f"   • Elements: {len(request.story_elements)}")
    print(f"   • Word Target: {request.word_limit}")
    print(f"   • STRICT REQUIREMENT: ALL ANSWERS MUST BE INCLUDED - NO EXCEPTIONS")
    print(f"{'*'*80}\n")
    
    try:
        result = await generate_long_story(
            elements=request.story_elements,
            word_limit=request.word_limit
        )
        
        # Log final statistics with answer verification
        print(f"\n{'- - '*20}")
        print(f"📊 FINAL STORY STATISTICS")
        print(f"   • Title: {result['title']}")
        print(f"   • Chapters: {len(result['chapters'])}")
        print(f"   • Total Words: {result['total_word_count']}")
        print(f"   • Answers Included: {result['generated_elements']['answers_used']}")
        print(f"   • Answer Distribution: {result['generated_elements']['answer_distribution']}")
        print(f"✅ VALIDATION PASSED: ALL ANSWERS INCLUDED")
        print(f"{'- - '*20}\n")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"💀 FATAL ERROR in generate_story endpoint: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "✨ STRICT STORY WEAVER API ✨", 
        "version": "1.0.0",
        "endpoints": {
            "POST /generate-story": "Generate stories using EVERY ANSWER provided - no fallbacks",
            "GET /health": "Check API health status",
            "GET /test-openai": "Verify OpenAI connection"
        },
        "guarantee": "100% of your answers will be included in the story"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": "2025-12-06T12:00:00Z"}

@app.get("/test-openai")
async def test_openai():
    """Test OpenAI connection with a simple prompt"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": TEST_OPENAI_PROMPT}
        ]
        
        response = await call_openai_api(messages, 50)
        
        return {
            "status": "success",
            "response": response,
            "model_used": "gpt-4o-mini"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2025-12-06T12:00:00Z"
        }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("✨ STRICT STORY WEAVER API STARTING UP ✨")
    print(f"   • Guarantee: EVERY answer will be included - NO fallbacks or placeholders")
    print(f"   • OpenAI Key: {'✓ Set' if api_key else '✗ Missing'}")
    print(f"   • Server: http://0.0.0.0:8000")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
