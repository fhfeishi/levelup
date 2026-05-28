


from importlib.metadata import version

core_version = version("langchain-core")
lg_version = version("langgraph") 





def main():
    print("Hello from pipelines-rag!")
    print(core_version)
    print(lg_version)


if __name__ == "__main__":
    main()
