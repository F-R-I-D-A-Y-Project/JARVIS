import pexpect
import platform


class Process:
    '''
        This class is responsible for the communication between the chatbot and the shell
    '''

    def __init__(self) -> None:
        self.platform = platform.system()
        if self.platform == 'Windows':
            self.init_cmd = "cmd.exe"
        elif self.platform == 'Linux' or self.platform == 'Darwin':
            self.init_cmd = "bash"
        self.child_process: pexpect.spawn

    def send(self, message: str | list[str]) -> None:
        '''
            This method sends a message to the shell and returns the output of the command.

            Args:
                message (str): The message or command to be sent to the shell.
        '''
        if isinstance(message, list):
            self.child_process.sendline('; '.join(message))
            self.child_process.expect_exact('; '.join(message))
        else:
            self.child_process.sendline(message)
            self.child_process.expect_exact(message)
        self.child_process.expect_exact('$')
        ret: str = self.child_process.before.decode("utf-8")
        try:
            ret = '\n'.join(ret.split('\n')[1: -2])
        except IndexError:
            ret = ''
        finally:
            return ret

    def __enter__(self) -> pexpect.spawn:
        self.child_process = pexpect.spawn(self.init_cmd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.child_process.close()


def main():
    with Process() as mcs:
        print(mcs.send(["ls -la", "pwd"]))


if __name__ == "__main__":
    main()
