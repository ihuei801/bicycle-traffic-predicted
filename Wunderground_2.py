import requests
import json
import csv
import time

#key = '4e569762648c374e'
key_2 = '95f21e06821e30f1'
rfile = '2016-Q3-cabi-trips-history-data/2016-Q3-Trips-History-Data-1.csv'
wfile = '2016-Q3-cabi-trips-history-data/2016-Q3-Trips-History-Data-1-weather_3.csv'
csvrfile = open(rfile, 'rb')
csvwfile = open(wfile, 'wb')
#url = 'http://api.wunderground.com/api/' + key_2 + '/history_20110101/q/DC.json'
reader = csv.DictReader(csvrfile)
rfieldnames = reader.fieldnames
wfieldnames = ['Start date', 'Weather type', 'Temperature', 'Humidity', 'Wind speed']
writer = csv.DictWriter(csvwfile, fieldnames=wfieldnames)
writer.writeheader()
visit = set()
for row in reader:
    d = {}
    t = row['Start date']
    t = t.split()
    date = t[0].split('/')
    year = int(date[2])
    mon = int(date[0])
    mday = int(date[1])
    date = str(year) + '%02d' % mon + str(mday)
    t = t[1].split(':')
    hour = int(t[0])
    key = date + ":" + str(hour)
    if key in visit:
        continue
    visit.add(key)
    #print date, hour
    get = False
    while not get:
        try:
            url = 'http://api.wunderground.com/api/' + key_2 + '/history_' + date + '/q/DC.json'
            r = requests.get(url)
            parsed_json = json.loads(r.text)
            observations = parsed_json["history"]["observations"]
            get = True
        except:
            time.sleep(100)
    for info in observations:

        if int(info["date"]["year"]) == year and int(info["date"]["mon"]) == mon and int(
                info["date"]["mday"]) == mday and int(info["date"]["hour"]) == hour:
            d['Weather type'] = info["icon"]
            d['Temperature'] = info["tempm"]
            d['Humidity'] = float(info["hum"]) / 100
            d['Wind speed'] = info["wspdm"]
            break

    d['Start date'] = row['Start date'].split(':')[0]
    writer.writerow(d)





