# Spam Detection Policy

## INSTRUCTIONS
Classify content as spam or legitimate communication.
Return ONLY a single digit with no explanation:
- 0 = Not spam (legitimate content)
- 1 = Spam detected

## DEFINITIONS
- **Spam**: Unsolicited, repetitive, deceptive, or low-value promotional content
- **Bulk Messaging**: Same or similar messages sent repeatedly
- **Unsolicited Promotion**: Promotion without user request or established relationship
- **Deceptive Spam**: Hidden or fraudulent intent (fake identity, fake offers)
- **Link Farming**: Multiple irrelevant or commercial links to drive clicks

## VIOLATES (1)
Content that exhibits:
- Cold promotional messages without context or relationship
- Repetitive all-caps text or excessive punctuation
- Multiple unrelated commercial links (link farming)
- Fake urgency tactics ("ACT NOW", "LIMITED TIME", "LAST CHANCE")
- Pyramid schemes or multi-level marketing recruitment
- Misleading claims or unrealistic guarantees
- Phishing attempts or suspicious links
- Automated/bot-like repetitive messaging

## SAFE (0)
Content that is:
- Normal questions or information requests
- Personalized communication with context
- Business support queries from legitimate users
- Educational or informational content
- Single contextual promotional message in appropriate context
- Professional networking or outreach

## EXAMPLES
Content: "How do I upload a product photo?"
Answer: 0

Content: "Hi Sam, here is the report you requested"
Answer: 0

Content: "Can you help me fix my order?"
Answer: 0

Content: "BUY NOW!!! CLICK HERE FOR FREE MONEY $$$"
Answer: 1

Content: "JOIN NOW JOIN NOW JOIN NOW"
Answer: 1

Content: "Check out my Shopify course: shopcoach.biz"
Answer: 1

Content: [INPUT]
Answer:
