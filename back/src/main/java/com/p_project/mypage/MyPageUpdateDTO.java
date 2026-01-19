package com.p_project.mypage;

import lombok.*;

import java.time.LocalDate;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MyPageUpdateDTO {

    private Long userId;
    private String email;
    private String nickName;
    private LocalDate birthDate;


}
