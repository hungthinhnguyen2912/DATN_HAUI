class UserModel {
  final String name;
  final String email;
  final DateTime createdAt;
  final String uid;
  String avatarUrl;

  UserModel({
    required this.uid,
    required this.name,
    required this.email,
    required this.createdAt,
    this.avatarUrl = "",
  });

  Map<String, dynamic> toJson() =>
      {
        "name": name,
        "email": email,
        "createdAt": createdAt,
        "uid": uid,
      };
}
