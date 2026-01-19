package com.p_project.mypage;

import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MyPageDTO {

    private Long userId;
    private String email;
    private String nickName;
    private String profileURL;
    private LocalDate birthDate;

    public void updateProfile(String nickname, String email, String profileURL, LocalDate birthDate) {
        this.nickName = nickname;
        this.email = email;
        this.profileURL = profileURL;
        this.birthDate = birthDate;
    }
}
