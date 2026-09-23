import re

def check_red_flags(text):
    """
    Scans a job description for common scam indicators.
    Returns a list of red flags found.
    """
    flags = []
    
    # Convert text to lowercase for easier matching
    text_lower = str(text).lower()

    # Rule 1: Free/Personal Email Domains
    if re.search(r'@(?:gmail|yahoo|hotmail|outlook|aol)\.com', text_lower):
        flags.append("Uses a free/personal email address instead of a company domain.")

    # Rule 2: Requests for Payment or Bank Details
    if re.search(r'\b(registration fee|processing fee|pay to apply|bank details|wire transfer|western union)\b', text_lower):
        flags.append("Mentions fees, payments, or asks for bank details.")

    # Rule 3: High Urgency Language
    if re.search(r'\b(urgently hiring|act fast|hiring immediately|immediate start)\b', text_lower):
        flags.append("Uses high-pressure or extreme urgency language.")

    # Rule 4: Unrealistic Promises / Too Good to be True
    if re.search(r'\b(no experience required|earn thousands|get rich|financial freedom)\b', text_lower):
        flags.append("Contains unrealistic promises or 'too good to be true' phrases.")

    return flags

# --------------------------------
# Test the Rule Engine
# --------------------------------
if __name__ == "__main__":
    sample_job = """
    Urgently hiring Data Entry Clerks! 
    No experience required. Earn thousands working from home! 
    Please send your resume to HRdept123@gmail.com. 
    Note: A $50 processing fee is required for background checks.
    """
    
    print("Testing Rule Engine...\n")
    found_flags = check_red_flags(sample_job)
    
    if found_flags:
        print(f"⚠️ Found {len(found_flags)} Red Flags:")
        for flag in found_flags:
            print(f" - {flag}")
    else:
        print("✅ No red flags detected.")