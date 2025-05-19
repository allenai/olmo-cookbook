from threading import current_thread, main_thread


def thread_rank() -> int:
    if (t := current_thread()) == main_thread():
        return 0

    _, name, *_ = t.name.split("-")
    return int(name)
