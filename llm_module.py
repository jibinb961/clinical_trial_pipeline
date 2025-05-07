import os
import google.generativeai as genai
from typing import List, Dict, Optional
from dotenv import load_dotenv
import pandas as pd
import datetime

# Load environment variables
load_dotenv()

# Configure Gemini API - use only one API key variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")
else:
    # Configure the Gemini API with the key
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini client with safety settings for research/medical content
def initialize_gemini():
    """
    Initialize the Gemini API client.
    
    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return True
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        return False

def format_trials_for_prompt(trials: List[Dict], sponsor_name: str) -> str:
    """
    Format clinical trial data for a structured LLM prompt.
    
    Args:
        trials: List of clinical trial data dictionaries
        sponsor_name: Name of the sponsor company
        
    Returns:
        Formatted prompt string for Gemini
    """
    prompt = f"""
The following clinical trial data is from {sponsor_name}. Each entry includes the disease being studied,
number of participants enrolled, timeline information, and a short summary of the study. Based on this, 
generate a concise summary of what {sponsor_name} is currently focusing on in their research pipeline.

Please include:
1. The main disease areas or therapeutic focuses
2. The stages of development (based on trial phases)
3. Timeline patterns (when trials started, expected completions)
4. Any notable trends or patterns in their research

CLINICAL TRIAL DATA:
"""
    
    # Add each trial to the prompt
    for i, trial in enumerate(trials, 1):
        # Format dates for display in prompt
        start_date = trial.get('start_date', 'N/A')
        completion_date = trial.get('completion_date', 'N/A') 
        primary_completion_date = trial.get('primary_completion_date', 'N/A')
        
        prompt += f"""
Trial {i}:
- NCT ID: {trial.get('nct_id', 'N/A')}
- Disease/Condition: {trial.get('conditions', 'N/A')}
- Enrollment: {trial.get('enrollment', 'N/A')} participants
- Phase: {trial.get('phase', 'N/A')}
- Status: {trial.get('status', 'N/A')}
- Timeline: Start: {start_date}, Primary Completion: {primary_completion_date}, Full Completion: {completion_date}
- Summary: {trial.get('brief_summary', 'N/A')[:300]}...
"""
    
    # Add final instructions
    prompt += """
Based on the clinical trial data above, provide a well-structured analysis of the sponsor's current research focus.
Format your response in clear paragraphs with headings for:
- MAIN DISEASE AREAS
- DEVELOPMENT STAGES
- TIMELINE PATTERNS
- RESEARCH TRENDS
"""
    
    return prompt

def generate_sponsor_analysis(trials: List[Dict], sponsor_name: str) -> Optional[str]:
    """
    Generate an analysis of a sponsor's clinical trial portfolio using Gemini AI.
    
    Args:
        trials: List of trial data dictionaries
        sponsor_name: Name of the sponsor company
        
    Returns:
        Generated analysis text
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
    
    if not trials:
        return f"No trials found for {sponsor_name} in the selected date range."
    
    # Create a summary of the trials for the prompt
    trials_summary = "\n\n".join([
        f"Trial ID: {trial.get('nct_id', 'N/A')}\n"
        f"Title: {trial.get('brief_title', 'N/A')}\n"
        f"Phase: {trial.get('phase', 'N/A')}\n"
        f"Status: {trial.get('status', 'N/A')}\n"
        f"Conditions: {trial.get('conditions', 'N/A')}\n"
        f"Start Date: {trial.get('start_date', 'N/A')}\n"
        f"Completion Date: {trial.get('completion_date', 'N/A')}\n"
        f"Enrollment: {trial.get('enrollment', 'N/A')}"
        for trial in trials
    ])
    
    # Create the prompt for Gemini
    prompt = f"""
    Please analyze the following clinical trial data for {sponsor_name} and provide insights about their research focus and strategy:

    {trials_summary}

    In your analysis, please include:
    1. Key therapeutic areas the company is focusing on
    2. Distribution of trials across different phases
    3. Notable patterns in trial size and scope
    4. Potential strategic priorities based on the trial portfolio
    5. Any emerging trends or shifts in research focus
    
    Format your response in markdown with clear sections and bullet points where appropriate.
    """
    
    try:
        # Generate content with Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Return the formatted analysis
        return response.text
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def parse_date_for_analysis(date_str):
    """
    Parse a date string from the API to a datetime object for stock analysis.
    
    Args:
        date_str: Date string in ISO 8601 format (yyyy, yyyy-MM, yyyy-MM-dd)
        
    Returns:
        datetime.datetime object or None if parsing fails
    """
    if not date_str:
        return None
        
    try:
        # Handle different ISO 8601 formats
        if len(date_str) == 4:  # yyyy
            return datetime.datetime.strptime(f"{date_str}-01-01", '%Y-%m-%d')
        elif len(date_str) == 7:  # yyyy-MM
            return datetime.datetime.strptime(f"{date_str}-01", '%Y-%m-%d')
        elif len(date_str) >= 10:  # yyyy-MM-dd or longer
            return datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
        else:
            return None
    except Exception:
        return None

def format_price(value):
    """Format price values with proper handling of None/N/A values"""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        return f"{value:.2f}"
    except (ValueError, TypeError):
        return 'N/A'

def format_percent(value):
    """Format percentage values with proper handling of None/N/A values"""
    if value is None or value == 'N/A':
        return 'N/A'
    try:
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        return 'N/A'

def generate_stock_correlation_analysis(ticker: str, trials: List[Dict], stock_data, start_date, end_date) -> str:
    """
    Generate an analysis of the correlation between clinical trial start dates and stock price movements.
    
    Args:
        ticker: Stock ticker symbol
        trials: List of trial data dictionaries
        stock_data: DataFrame with stock price data
        start_date: Start date of the analysis period
        end_date: End date of the analysis period
        
    Returns:
        Generated analysis text
    """
    try:
        print(f"Starting analysis for {ticker} with {len(trials)} trials")
        
        if not GEMINI_API_KEY:
            print("Error: No Gemini API key found")
            return "Error: Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
        
        if not trials or stock_data is None or stock_data.empty:
            print(f"Insufficient data for {ticker}: trials={len(trials) if trials else 0}, stock_data_empty={stock_data.empty if stock_data is not None else True}")
            return f"Insufficient data to analyze correlation for {ticker}."
        
        # Extract price column name based on what's available
        is_multiindex = isinstance(stock_data.columns, pd.MultiIndex)
        if is_multiindex:
            print(f"Stock data has MultiIndex columns: {stock_data.columns.tolist()[:5]}")
            if ('Adj Close', ticker) in stock_data.columns:
                price_column = ('Adj Close', ticker)
            else:
                price_column = ('Close', ticker)
        else:
            print(f"Stock data has standard columns: {stock_data.columns.tolist()[:5]}")
            price_column = 'Adj Close' if 'Adj Close' in stock_data.columns else 'Close'
        
        print(f"Using price column: {price_column}")
        
        # Define the window size for price comparison (increased from 5 to 15 days)
        window_days = 15
        
        # Create a summary of trial start dates and nearby stock prices
        trial_price_data = []
        for trial in trials:
            start_date_str = trial.get('start_date')
            if start_date_str:
                # Convert to datetime
                trial_date = parse_date_for_analysis(start_date_str)
                
                if trial_date:
                    # Get stock prices before and after the trial start date
                    try:
                        # Find nearest date indices
                        idx = stock_data.index.get_indexer([trial_date], method='nearest')[0]
                        
                        if idx >= 0 and idx < len(stock_data):
                            # Get price on the trial date
                            price_on_date = stock_data.iloc[idx][price_column]
                            if isinstance(price_on_date, pd.Series):
                                price_on_date = price_on_date.iloc[0]
                                
                            # Get price 15 days before if available
                            price_before = None
                            if idx >= window_days:
                                price_before = stock_data.iloc[idx-window_days][price_column]
                                if isinstance(price_before, pd.Series):
                                    price_before = price_before.iloc[0]
                            
                            # Get price 15 days after if available
                            price_after = None
                            if idx + window_days < len(stock_data):
                                price_after = stock_data.iloc[idx+window_days][price_column]
                                if isinstance(price_after, pd.Series):
                                    price_after = price_after.iloc[0]
                            
                            # Calculate percent changes
                            before_change = None
                            if price_before is not None:
                                before_change = ((price_on_date - price_before) / price_before) * 100
                                
                            after_change = None
                            if price_after is not None:
                                after_change = ((price_after - price_on_date) / price_on_date) * 100
                            
                            trial_price_data.append({
                                "nct_id": trial.get('nct_id'),
                                "title": trial.get('brief_title'),
                                "start_date": start_date_str,
                                "price_on_date": price_on_date,
                                "price_15d_before": price_before,
                                "price_15d_after": price_after,
                                "pct_change_before": before_change,
                                "pct_change_after": after_change
                            })
                    except Exception as e:
                        print(f"Error processing trial {trial.get('nct_id')}: {str(e)}")
        
        print(f"Processed {len(trial_price_data)} trials with price data")
        
        if not trial_price_data:
            return "No trials with valid price data were found in the selected period. This could be because trial start dates don't overlap with trading days or stock data is unavailable for those dates."
        
        # Format the trial price data for the prompt
        trial_price_summary = "\n\n".join([
            f"Trial ID: {data.get('nct_id', 'N/A')}\n"
            f"Title: {data.get('title', 'N/A')}\n"
            f"Start Date: {data.get('start_date', 'N/A')}\n"
            f"Price on Start Date: ${format_price(data.get('price_on_date'))}\n"
            f"Price 15 Days Before: ${format_price(data.get('price_15d_before'))}\n"
            f"Price 15 Days After: ${format_price(data.get('price_15d_after'))}\n"
            f"% Change 15 Days Before: {format_percent(data.get('pct_change_before'))}\n"
            f"% Change 15 Days After: {format_percent(data.get('pct_change_after'))}"
            for data in trial_price_data
        ])
        
        # Get overall stats for the entire period
        overall_start_price = stock_data.iloc[0][price_column]
        if isinstance(overall_start_price, pd.Series):
            overall_start_price = overall_start_price.iloc[0]
            
        overall_end_price = stock_data.iloc[-1][price_column]
        if isinstance(overall_end_price, pd.Series):
            overall_end_price = overall_end_price.iloc[0]
            
        overall_change = ((overall_end_price - overall_start_price) / overall_start_price) * 100
        
        # Create the prompt for Gemini
        prompt = f"""
        Please analyze the correlation between clinical trial start dates and stock price movements for {ticker} during the period from {start_date} to {end_date}.

        Overall Stock Performance:
        - Starting Price: ${format_price(overall_start_price)}
        - Ending Price: ${format_price(overall_end_price)}
        - Overall Change: {format_percent(overall_change).replace('%', '')}%

        Clinical Trial Start Dates and Stock Prices:
        {trial_price_summary}

        In your analysis, please include:
        1. Assessment of whether there appears to be a correlation between trial start dates and stock price movements
        2. Identification of any trials that coincided with significant price changes (looking at the 15-day window before and after)
        3. Analysis of whether stock prices generally increase, decrease, or remain stable after trial start announcements
        4. Overall interpretation of how this company's clinical trial pipeline appears to influence its stock performance
        5. Potential factors beyond clinical trials that might be affecting the stock price during this period
        
        Format your response in markdown with clear sections and bullet points where appropriate.
        """
        
        print("Prompt prepared, calling Gemini API...")
        
        try:
            # Generate content with Gemini
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                }
            ]
            
            model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings)
            
            # Set timeout and retry
            generation_config = {
                "temperature": 0.4,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
            
            # Make the API call with explicit configuration
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=False
            )
            
            print("Received response from Gemini API")
            
            # Check if response has the expected attributes
            if not hasattr(response, 'text'):
                print(f"Warning: Response doesn't have 'text' attribute. Response type: {type(response)}")
                return "Error: Unexpected response format from Gemini API. Please try again."
            
            # Return the formatted analysis
            return response.text
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            print(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error in correlation analysis: {str(e)}"
        print(error_msg)
        return error_msg 