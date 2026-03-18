import os


def main():
    while True:
        print("\nVälj agent:\n")
        print("1. Simple Agent")
        print("2. Tool Agent")
        print("3. RAG Agent")
        print("0. Avsluta")

        choice = input("\nSkriv ditt val (1/2/3/0): ")

        if choice == "1":
            print("\nStartar Simple Agent...\n")
            os.system("python3 -m examples.agent_lecture.agent_1")

        elif choice == "2":
            print("\nStartar Tool Agent...\n")
            os.system("python3 -m examples.agent_lecture.agent_2")

        elif choice == "3":
            print("\nStartar RAG Agent...\n")
            os.system("python3 -m examples.agent_lecture.agent_3")

        elif choice == "0":
            print("Avslutar programmet.")
            break

        else:
            print("Ogiltigt val, försök igen.")


if __name__ == "__main__":
    main()