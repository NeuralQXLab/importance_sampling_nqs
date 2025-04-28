# Base class for all wrappers, has save/load functionality implemented
import json


class NetBase:
    def name_and_arguments_to_dict(self):
        return {"Name": "NetBase"}

    @staticmethod
    def argument_saver(
        arg_dict: dict, file_name: str, prefix: str = None, write_mode: str = "a"
    ):
        """
        Save the arg_dict to file fname
        """
        if not prefix == None:
            save_dict = {prefix: arg_dict}
        else:
            save_dict = arg_dict

        with open(file_name, write_mode) as f:
            json.dump(save_dict, f)

    def save(self, file_name: str, prefix: str = None, write_mode: str = "a"):
        """
        Save the network arguments
        """
        arg_dict = self.name_and_arguments_to_dict()
        self.argument_saver(arg_dict, file_name, prefix, write_mode)

    @staticmethod
    def argument_loader(file_name: str, prefix: str = None):
        """
        Load in the dictionary of arguments
        """
        with open(file_name, "r") as f:
            load_dict = json.load(f)

        if not prefix == None:
            out_dict = load_dict[prefix]
        else:
            out_dict = load_dict

        return out_dict
