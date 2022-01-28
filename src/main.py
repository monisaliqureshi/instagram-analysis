from fastapi import FastAPI, Request
from instagramy import InstagramUser
import face_recognition as fr
import urllib.request
from deepface.DeepFace import analyze
from config import client
import uvicorn
from detection import apply_filter, make_desc



mycol = client['IG']['IGcollection']

def check_image(url):
    response = urllib.request.urlopen(url)
    image = fr.load_image_file(response)
    objects = apply_filter(image)
    description = make_desc(objects)
    faces = fr.face_locations(image)
    result = dict()
    result['Total Faces'] = len(faces)
    result['description'] = description
    if len(faces)==0:
        return result
    index = 1
    for top, right, bottom, left in faces:
        face_image = image[top:bottom, left:right]
        analysis = analyze(face_image, actions=['emotion'], enforce_detection=False)
        result["face" + str(index)] = analysis
    return result


app = FastAPI()

@app.post("/get_data")
async def get_post(request: Request):

    data = await request.json()
    uname = data['username']
    max_post = int(data['max_post'])
    user = InstagramUser(uname)
    u_data = dict()
    u_data['username'] = uname
    u_data['biography'] = user.biography
    u_data['is_verified'] = user.is_verified
    index = 1
    for post in user.posts:
        if post.is_video:
            continue
        post_data = dict()
        caption = post.caption
        post_data['caption'] = caption
        post_data['post_url'] = post.post_url
        post_data['post_source'] = post.post_source
        image_result = check_image(post.post_source)
        post_data['image_result'] = image_result
        u_data['post' + str(index)] = post_data
        if max_post == index:
            break
        index += 1
    x = mycol.insert_one(u_data)
    return {"res": "Done"}


if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=4000)