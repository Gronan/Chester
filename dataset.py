from datatables import TableAbstract


def tables_factory(parameters: dict) -> dict[TableAbstract]:
    tables = {}
    # generate a dataframe for each tables in parameters dictionary
    for table_name, table_parameters in parameters.items():
        table = TableAbstract(table_name, table_parameters)
        tables[table_name] = table

    return tables


class Dataset:
    name: str
    tables: dict[TableAbstract]

    def __init__(self, parameters):
        self.name = parameters['name']
        self.tables = tables_factory(parameters["tables"])
        self.build_tables()

    def build_tables(self) -> None:
        for table_name, table in self.tables.items():
            if table.is_generated():
                continue
            else:
                table.generate_data()
