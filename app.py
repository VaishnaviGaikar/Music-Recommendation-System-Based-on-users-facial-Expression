from flask import Flask, render_template, Response, jsonify
import gunicorn
from camera import *

app = Flask(__name__)

headings = ("Name", "Album", "Artist")
df1 = music_rec()
df1 = df1.head(15)
image = None
emotions = {}    


@app.route('/')
def index():
    print(f"Inside Index : {df1.to_json(orient='records')}")
    return render_template('index.html', headings=headings, data=df1)


def gen(camera):
    from deepface import DeepFace
    global emotions
    while True:
        global df1
        global image
        frame, df1, image = camera.get_frame()
        if image is not None:
            print(f"Potential Image Found : {type(image)}")
            try:
                emotion_analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
                emotions = emotion_analysis[0].get('emotion')
            except Exception as e:
                print(f'Error: Emotion analysis error - {str(e)}')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    global image
    print(f"Video-Feed called")
    vf = gen(VideoCamera())
    # emotion_analysis = DeepFace.analyze(image, actions = ['emotion'])
    # print(emotion_analysis)
    return Response(vf,
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')


@app.route('/emotion_analysis')
def get_emotion_analysis():
    emotion_tag = list(emotions.keys())
    emotion_val = list(emotions.values())
    dominant_emotion_val = emotion_val[emotion_val.index(max(emotion_val))]
    dominant_emotion_tag = emotion_tag[emotion_val.index(max(emotion_val))]
    return {
        "Tag" : dominant_emotion_tag,
        "Value" : dominant_emotion_val
    }

if __name__ == '__main__':
    app.debug = True
    app.run()
