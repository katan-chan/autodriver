import urllib.request
import urllib.error

def test_cors():
    url = "http://localhost:8000/config"
    origin = "http://192.168.1.67:8001"
    
    req = urllib.request.Request(url)
    req.add_header("Origin", origin)
    
    print(f"Testing CORS for origin: {origin}")
    try:
        with urllib.request.urlopen(req) as response:
            print(f"Status: {response.status}")
            headers = response.info()
            aca_origin = headers.get("Access-Control-Allow-Origin")
            aca_credentials = headers.get("Access-Control-Allow-Credentials")
            
            print(f"Access-Control-Allow-Origin: {aca_origin}")
            print(f"Access-Control-Allow-Credentials: {aca_credentials}")
            
            if aca_origin == origin and aca_credentials == "true":
                print("SUCCESS: CORS headers are correct.")
            else:
                print("FAILURE: CORS headers are missing or incorrect.")
                
    except urllib.error.URLError as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_cors()
