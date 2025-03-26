import 'package:cloud_firestore/cloud_firestore.dart';

class History {

  final String kind;
  final String imageUrl;
  final String uidUser;
  final Timestamp createdAt;

  History({required this.kind, required this.imageUrl, required this.uidUser, required this.createdAt});

  Map<String, dynamic> toJson() =>
      {
        "kind": kind,
        "imageUrl": imageUrl,
        "uidUser": uidUser,
        "createdAt": createdAt,
      };

}