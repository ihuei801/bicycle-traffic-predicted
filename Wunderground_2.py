import requests
import json
import csv
import time

#key = '4e569762648c374e'
key_2 = '95f21e06821e30f1'
years = ['2014']
seasons = ['Q4']

for y in years:
    for s in seasons:
        logfile = "weather/log_" + y + "_" + s + ".txt"
        log = open(logfile, "wb")
        rfile = 'data/' + y + '-' + s + '-cabi-trip-history-data.csv'
        wfile = 'weather/' + rfile.split('/')[1].split('.')[0] +  '-weather.csv'
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
            if '/' not in row['Start date']:
                date_and_time = row['Start date'].split()
                mdate = date_and_time[0]
                num = mdate.split('-')
                mdate = num[1] + '/' + num[2] + '/' + num[0]
                mtime = date_and_time[1]
                row['Start date'] = mdate + ' ' + mtime
            t = row['Start date']
            t = t.split()

            try:
                date = t[0].split('/') if '/' in t[0] else t[0].split('-')
                year = int(date[2])
                mon = int(date[0])
                mday = int(date[1])
                date = str(year) + '%02d' % mon + '%02d' % mday
                t = t[1].split(':')
                hour = int(t[0])
                key = date + ":" + str(hour)
                if key in visit:
                    continue
                visit.add(key)
            except:
                print row['Start date']
                continue
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
                    try:
                        d['Weather type'] = info["icon"]
                        d['Temperature'] = info["tempm"]
                        d['Humidity'] = float(info["hum"]) / 100
                        d['Wind speed'] = info["wspdm"]
                    except:
                        print year, mon, mday, hour
                        print info["icon"]
                        print info["tempm"]
                        print info["hum"]
                        print info["wspdm"]
                    break
                # else:
                    # if int(info["date"]["hour"]) != hour:
                    #     print "hour not equal"
                    #     print int(info["date"]["hour"])
                    #     print hour
                    # if int(info["date"]["year"]) != year:
                    #     print "year not equal"
                    # if int(info["date"]["mon"]) != mon:
                    #     print "mon not equal"
                    # if int(info["date"]["mday"]) != mday:
                    #     print "mday not equal"
            d['Start date'] = row['Start date'].split(':')[0]
            if len(d) != 5:
                print date + ":" + str(hour)
                log.write(date + ":" + str(hour)+ "\n")
            else:
                writer.writerow(d)
        print y + "-" + s + " finished"
        log.write(y + "-" + s + " finished\n")



