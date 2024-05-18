from flask import Flask


s1='red'
s2='red'
s3='red'
s4='red'
s5='red'
x=0
y=0

app = Flask(__name__)
@app.route('/')
def root():
    return {
            'semaphore_1':s1,
            'semaphore_2':s2,
            'semaphore_3':s3,
            'semaphore_4':s4,
            'semaphore_5':s5,
            'loc_x':x,
            'loc_y':y
            }


@app.route('/s1/<val>')
def api_s1(val):
    global s1
    s1=val
    return 'success'

@app.route('/s2/<val>')
def api_s2(val):
    global s2
    s2=val
    return 'success'

@app.route('/s3/<val>')
def api_s3(val):
    global s3
    s3=val
    return 'success'

@app.route('/s4/<val>')
def api_s4(val):
    global s4
    s4=val
    return 'success'

@app.route('/s5/<val>')
def api_s5(val):
    global s5
    s5=val
    return 'success'

@app.route('/x/<val>')
def api_x(val):
    global x
    x=val
    return 'success'

@app.route('/y/<val>')
def api_y(val):
    global y
    y=val
    return 'success'

if __name__=='__main__':
    app.run(port=5901)


