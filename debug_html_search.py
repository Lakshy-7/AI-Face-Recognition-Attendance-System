import urllib.request

html = urllib.request.urlopen('http://127.0.0.1:7868').read().decode(errors='ignore')
for name in ['get_registered_students', 'load_student_options', 'load_attendance_frame', 'update_student_profile']:
    print(name, name in html, html.count(name))
