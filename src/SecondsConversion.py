while True:
	sec = int(raw_input("Enter amount of seconds: "))
	hour = sec // 3600
	sec = sec % 3600
	minute = sec // 60
	sec = sec % 60
	print str(hour) + ":" + str(minute) + ":" + str(sec)