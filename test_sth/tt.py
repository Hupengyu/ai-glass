pid_info_path = 'test.txt'


def pid_read(pid_info_path):
    dir_set = set()
    with open(pid_info_path, 'r', encoding='utf-8') as f_r:
        dirs = f_r.readlines()
        if len(dirs) > 0:
            for dir in dirs:
                print('dir: ', dir)
                dir_set.add(dir)
                print('dir_set: ', dir_set)


if __name__ == '__main__':
    pid_read(pid_info_path)
