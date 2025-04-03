import 'package:cloud_firestore/cloud_firestore.dart';

class History {
  final String kind;
  final String imageUrl;
  final String uidUser;
  final Timestamp createdAt;
  final String publicId;

  History({
    required this.publicId,
    required this.kind,
    required this.imageUrl,
    required this.uidUser,
    required this.createdAt,
  });

  Map<String, dynamic> toJson() => {
    "kind": kind,
    "imageUrl": imageUrl,
    "uidUser": uidUser,
    "createdAt": createdAt,
    "publicId": publicId,
  };

  factory History.fromJson(Map<String, dynamic> json) => History(
    kind: json['kind'],
    imageUrl: json['imageUrl'],
    uidUser: json['uidUser'],
    createdAt: json['createdAt'],
    publicId: json['publicId']
  );
}
