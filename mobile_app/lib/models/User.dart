import 'package:cloud_firestore/cloud_firestore.dart';

class UserModel {
  final String name;
  final String email;
  final Timestamp createdAt;
  final String uid;
  String avatarUrl;
  String publicIdAvatar;


  UserModel({
    required this.uid,
    required this.name,
    required this.email,
    required this.createdAt,
    this.avatarUrl = "",this.publicIdAvatar = "",
  });

  Map<String, dynamic> toJson() =>
      {
        "name": name,
        "email": email,
        "createdAt": createdAt,
        "uid": uid,
        "avatarUrl": avatarUrl,
        "publicIdAvatar": publicIdAvatar,
      };
}

// class UserModel {
//   UserModel({
//     required this.uid,
//     required this.name,
//     required this.email,
//     required this.createdAt,
//     required this.image,
//     this.avatar,
//   });
//
//   final String uid;
//   final String name;
//   final double email;
//   final String createdAt;
//   final String image;
//    Avatar avatar;
//
//   factory UserModel.fromJson(Map<String, dynamic> json){
//     return UserModel(
//       uid: json["uid"],
//       name: json["name"],
//       email: json["email"],
//       createdAt: json["createdAt"],
//       image: json["image"],
//       avatar: json["avatar"] == "" ? "" : Avatar.fromJson(json["avatar"]),
//     );
//   }
//
//   Map<String, dynamic> toJson() => {
//     "uid": uid,
//     "name": name,
//     "email": email,
//     "createdAt": createdAt,
//     "image": image,
//     "avatar": avatar?.toJson(),
//   };
//
// }
//
// class Avatar {
//   Avatar({
//      this.avatarUrl = "",
//      this.publicId = "",
//   });
//
//    String avatarUrl;
//    String publicId;
//
//   factory Avatar.fromJson(Map<String, dynamic> json){
//     return Avatar(
//       avatarUrl: json["avatarUrl"],
//       publicId: json["public_id"],
//     );
//   }
//
//   Map<String, dynamic> toJson() => {
//     "avatarUrl": avatarUrl,
//     "public_id": publicId,
//   };
//
// }

