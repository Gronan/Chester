from mask import MaskAbstract


class TableAbstract:
    name: str
    type: str
    length: int
    gen_method: str
    tables_dependencies: list
    is_generated: bool
    masks: MaskAbstract
    parameters: dict

    def __init__(self, table_name: str, table_parameters: dict):
        self.name = table_name
        self.parameters = table_parameters

    def is_generated(self):
        return self.is_generated

    # compute using list comprehension
    def are_dependencies_generated(self) -> bool:
        return all([table.is_generated() for table in self.tables_dependencies])

    def generate_data(self) -> None:
        if self.are_dependencies_generated():
            pass
        else:
            for dependency in self.tables_dependencies:
                if dependency.is_generated():
                    continue
                dependency.generate_data()



