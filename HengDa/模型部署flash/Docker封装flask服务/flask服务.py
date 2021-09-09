# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/Project')
# def Project():
#     return '来了？老哥儿！给个关注&点赞不迷路哟'
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask
app=Flask(__name__)
@app.route('/')
def hello():
    return 'hello world'
if __name__ == '__main__':
    app.run()