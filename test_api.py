import requests
import json

# Test articles
test_cases = [
    {
        "text": "Breaking: Scientists discover miracle cure that doctors dont want you to know about!",
        "expected": "Fake"
    },
    {
        "text": "The Federal Reserve announced today that interest rates will remain unchanged following their latest meeting.",
        "expected": "Real"
    },
    {
        "text": "You won't believe what this celebrity said! Click here now!",
        "expected": "Fake"
    }
]

print("="*60)
print("TESTING FAKE NEWS DETECTOR API")
print("="*60)

for i, case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {case['expected']} news")
    print(f"Text: {case['text'][:80]}...")
    
    response = requests.post(
        'http://127.0.0.1:5000/predict',
        json={'text': case['text']}
    )
    
    result = response.json()
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Status: {'✓ Correct' if result['label'] == case['expected'] else '✗ Wrong'}")

print("\n" + "="*60)
