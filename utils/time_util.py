def get_time_taken_str(time_taken):
    m, s = divmod(time_taken, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)