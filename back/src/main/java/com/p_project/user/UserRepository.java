package com.p_project.user;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<UserEntity, Integer> {
    // 필요한 쿼리 메서드를 추가 가능 (예: findByNickname(String nickname))
}