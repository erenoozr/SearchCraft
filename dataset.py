"""SearchCraft validation dataset: 10 technical documents (doc_id = list index + 1)."""

DOCUMENTS = [
    {   # 1
        "title": "Introduction to Machine Learning with Python",
        "tags": ["ml", "python"],
        "text": "Machine learning is the science of algorithms that extract patterns from data. Thanks to "
                "scikit-learn and pandas, Python is the most popular language for machine learning "
                "projects. In this guide we build regression and classification models with Python.",
    },
    {   # 2
        "title": "Enterprise Machine Learning with Java",
        "tags": ["ml", "java"],
        "text": "The Java ecosystem offers Weka and Deeplearning4j for machine learning applications. In "
                "enterprise systems, Java acts as a bridge that brings Python models into production.",
    },
    {   # 3
        "title": "Deep Learning with PyTorch",
        "tags": ["dl", "python"],
        "text": "Deep learning is a subfield of machine learning built on multi-layer neural networks. "
                "PyTorch is a Python library with a dynamic computation graph. Training deep learning "
                "models on a GPU is easy with PyTorch.",
    },
    {   # 4
        "title": "Image Classification with TensorFlow and Keras",
        "tags": ["dl"],
        "text": "TensorFlow is a deep learning framework developed by Google. With the Keras API you "
                "build convolutional neural networks for image classification. Deep learning models "
                "need large datasets.",
    },
    {   # 5
        "title": "Relational Database Design and SQL",
        "tags": ["db", "sql"],
        "text": "Relational database tables are designed with normalization rules. SQL queries with JOIN, "
                "GROUP BY and proper indexes determine relational database performance. PostgreSQL and "
                "MySQL are popular systems.",
    },
    {   # 6
        "title": "NoSQL Databases: MongoDB and Redis",
        "tags": ["db", "nosql"],
        "text": "A NoSQL database offers schemaless document and key-value models. MongoDB is "
                "document-oriented, while Redis is an in-memory database. Instead of SQL they expose "
                "their own query APIs.",
    },
    {   # 7
        "title": "Container-Based Deployment with Docker",
        "tags": ["devops"],
        "text": "Docker packages applications together with their dependencies into container images. "
                "Images are built from a Dockerfile, so deployment behaves the same in every environment. "
                "Python services can also run in Docker containers.",
    },
    {   # 8
        "title": "Scalable Deployment with Kubernetes",
        "tags": ["devops"],
        "text": "Kubernetes is an open-source platform for container orchestration. Pods, Deployments and "
                "Services provide autoscaling and zero-downtime deployment. Kubernetes clusters run "
                "Docker images.",
    },
    {   # 9
        "title": "Building REST APIs with Java Spring Boot",
        "tags": ["backend", "java"],
        "text": "Spring Boot makes it fast to build REST APIs with Java. It provides dependency injection, "
                "database access through JPA and an embedded server. Java is still very common in "
                "microservice architectures.",
    },
    {   # 10
        "title": "Natural Language Processing and Information Retrieval",
        "tags": ["nlp", "ir"],
        "text": "Natural language processing lets computers understand text. Search engines use inverted "
                "indexes and ranking functions such as BM25. With Python you can build tokenizers, "
                "stemmers and machine learning based text classifiers.",
    },
]
