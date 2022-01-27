import requests
print("-"*20)
print("Instagram Scraper")
print("-"*20)
while True:
    print("Press Ctrl+C to close the program..")
    uname = str(input("Enter username: "))
    max_post = int(input("Max post to fetch : "))
    data = {"username": uname, "max_post": max_post}
    res = requests.post("http://localhost:4000/get_data", json=data).json()
    print(res['res'])
    print("-"*20)